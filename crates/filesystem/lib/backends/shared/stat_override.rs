//! Stat virtualization via the `user.msb.override_stat` extended attribute.
//!
//! The host process runs unprivileged and cannot `chown`, create device nodes,
//! or set xattrs on symlinks (Linux). All ownership/permissions/type information
//! is stored in a 20-byte binary xattr that [`patched_stat`] applies on top of
//! the real host `stat`.
//!
//! ## Format
//!
//! The xattr stores a fixed-size 20-byte `#[repr(C, packed)]` struct with version byte,
//! uid, gid, mode (including file type bits S_IFMT), and rdev. Reading/writing is a single
//! `memcpy` — no text parsing needed. Unknown version bytes trigger `EIO` (hard fail).
//!
//! ## Linux Symlink Exception
//!
//! Real symlinks on Linux cannot have `user.*` xattrs on most filesystems. `patched_stat`
//! skips the xattr read for real host symlinks (detected via `S_IFLNK` in the unpatched stat).
//! File-backed symlinks (regular files with S_IFLNK in xattr) are handled normally.

use std::{
    ffi::CStr,
    io,
    os::fd::RawFd,
    sync::{Arc, Mutex, OnceLock},
};

use crate::stat64;

use super::platform;

//--------------------------------------------------------------------------------------------------
// Constants
//--------------------------------------------------------------------------------------------------

/// Xattr key for the microsandbox override stat, as a null-terminated C string.
pub(crate) const OVERRIDE_XATTR_KEY: &CStr = c"user.msb.override_stat";

/// Xattr key used to probe whether the host mount supports microsandbox xattrs.
const PROBE_XATTR_KEY: &CStr = c"user.msb._probe";

/// Current version of the binary override format.
const OVERRIDE_VERSION: u8 = 1;

/// Size of the binary override struct.
const OVERRIDE_SIZE: usize = std::mem::size_of::<OverrideStat>();

/// Maximum length of `/proc/self/fd/{fd}\0` for any non-negative i32 fd.
#[cfg(target_os = "linux")]
const PROC_SELF_FD_PATH_BUF_LEN: usize = 32;

//--------------------------------------------------------------------------------------------------
// Types
//--------------------------------------------------------------------------------------------------

/// Binary layout of the override xattr value (20 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(crate) struct OverrideStat {
    pub version: u8,
    pub _pad: [u8; 3],
    pub uid: u32,
    pub gid: u32,
    pub mode: u32,
    pub rdev: u32,
}

/// Guest-visible ownership mapping for host-backed user volume files.
///
/// The map is applied only when stat virtualization is enabled and no
/// per-file override xattr exists. Files owned by the host user running
/// microsandbox appear as the sandbox's effective guest user; files owned by
/// other host users appear as the overflow identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindIdentityMap {
    /// Host uid that owns ordinary caller-created files.
    pub host_owner_uid: u32,

    /// Guest uid to show for host-owner files.
    pub guest_uid: u32,

    /// Guest gid to show for host-owner files.
    pub guest_gid: u32,

    /// Guest uid to show for files not owned by [`host_owner_uid`](Self::host_owner_uid).
    pub overflow_uid: u32,

    /// Guest gid to show for files not owned by [`host_owner_uid`](Self::host_owner_uid).
    pub overflow_gid: u32,
}

/// Shared bind identity map that can be installed after guest user resolution.
pub type BindIdentityMapHandle = Arc<OnceLock<BindIdentityMap>>;

/// An inode's patched owner and mode, reused while the host stat they were patched from keeps
/// its ctime, which saves reading the override xattr on every lookup. Writing an xattr changes
/// the host ctime (checked on APFS), so an unchanged ctime means an unchanged override.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Default)]
pub(crate) struct PatchedStatCache(Mutex<Option<CachedPatch>>);

#[derive(Clone, Copy, PartialEq)]
struct PatchFields {
    uid: u32,
    gid: u32,
    mode: u32,
    rdev: i64,
}

#[derive(Clone, Copy)]
struct CachedPatch {
    ctime: (i64, i64),
    host: PatchFields,
    patched: PatchFields,
    /// The identity map is installed once, possibly after boot, and changes the patched owner.
    mapped: bool,
}

//--------------------------------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------------------------------

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl PatchedStatCache {
    /// `host` patched as stored, if it was stored from a host stat with the same ctime and the
    /// fields the override replaces.
    pub(crate) fn get(
        &self,
        host: &stat64,
        bind_identity_map: Option<&BindIdentityMapHandle>,
    ) -> Option<stat64> {
        let cached = (*self.0.lock().unwrap())?;
        if cached.mapped != is_mapped(bind_identity_map)
            || cached.ctime != ctime(host)
            || cached.host != PatchFields::of(host)
        {
            return None;
        }
        let mut st = *host;
        st.st_uid = cached.patched.uid;
        st.st_gid = cached.patched.gid;
        st.st_mode = cached.patched.mode as _;
        st.st_rdev = cached.patched.rdev as _;
        Some(st)
    }

    pub(crate) fn store(
        &self,
        host: &stat64,
        patched: &stat64,
        bind_identity_map: Option<&BindIdentityMapHandle>,
    ) {
        *self.0.lock().unwrap() = Some(CachedPatch {
            ctime: ctime(host),
            host: PatchFields::of(host),
            patched: PatchFields::of(patched),
            mapped: is_mapped(bind_identity_map),
        });
    }
}

impl PatchFields {
    // st_mode is u16 and st_rdev i32 on macOS.
    #[allow(clippy::unnecessary_cast)]
    fn of(st: &stat64) -> Self {
        Self {
            uid: st.st_uid,
            gid: st.st_gid,
            mode: st.st_mode as u32,
            rdev: st.st_rdev as i64,
        }
    }
}

impl BindIdentityMap {
    /// Create a map from one host owner uid to one guest uid/gid pair.
    pub fn new(host_owner_uid: u32, guest_uid: u32, guest_gid: u32) -> Self {
        Self {
            host_owner_uid,
            guest_uid,
            guest_gid,
            overflow_uid: 65534,
            overflow_gid: 65534,
        }
    }

    /// Create a map that presents every host owner as one guest uid/gid pair.
    ///
    /// Explicit per-mount ownership is a fallback for every inode without a
    /// persistent override, including host files owned by another account.
    /// Keeping both branches equal avoids leaking the host ownership split into
    /// the guest as the overflow identity.
    pub fn fixed(guest_uid: u32, guest_gid: u32) -> Self {
        Self {
            host_owner_uid: 0,
            guest_uid,
            guest_gid,
            overflow_uid: guest_uid,
            overflow_gid: guest_gid,
        }
    }

    /// Apply this map to a stat result in place.
    pub fn apply(&self, st: &mut stat64) {
        if st.st_uid == self.host_owner_uid {
            st.st_uid = self.guest_uid;
            st.st_gid = self.guest_gid;
        } else {
            st.st_uid = self.overflow_uid;
            st.st_gid = self.overflow_gid;
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Functions
//--------------------------------------------------------------------------------------------------

/// Read the override xattr and patch the given stat with virtualized fields.
///
/// If no override xattr is present, returns the stat unmodified.
/// Returns `EIO` if the xattr is corrupt (wrong version, short read).
pub(crate) fn patched_stat(
    fd: RawFd,
    st: stat64,
    xattr_enabled: bool,
    strict: bool,
    bind_identity_map: Option<&BindIdentityMapHandle>,
) -> io::Result<stat64> {
    if !xattr_enabled {
        return Ok(st);
    }

    // Real symlinks on host cannot have user xattrs on Linux.
    #[cfg(target_os = "linux")]
    if st.st_mode & libc::S_IFMT == libc::S_IFLNK {
        let mut st = st;
        apply_bind_identity_map(&mut st, bind_identity_map);
        return Ok(st);
    }

    apply_override(st, read_override(fd, strict), bind_identity_map)
}

/// Like [`patched_stat`], reading the override from `path` without following a final symlink
/// and without opening the file. On macOS a lookup names the inode by `/.vol/<dev>/<ino>`, so
/// this saves the open and close that `patched_stat` needs a descriptor for.
#[cfg(target_os = "macos")]
pub(crate) fn patched_stat_at_path(
    path: &CStr,
    mut st: stat64,
    xattr_enabled: bool,
    strict: bool,
    bind_identity_map: Option<&BindIdentityMapHandle>,
) -> io::Result<stat64> {
    if !xattr_enabled {
        return Ok(st);
    }
    match read_override_at_path(path, strict) {
        // The inode can't be read through its path; patched_stat's callers fall back the same
        // way when they can't open it.
        Err(e) if matches!(e.raw_os_error(), Some(libc::EACCES) | Some(libc::EPERM)) => {
            apply_bind_identity_map(&mut st, bind_identity_map);
            Ok(st)
        }
        result => apply_override(st, result, bind_identity_map),
    }
}

fn apply_override(
    mut st: stat64,
    result: io::Result<Option<OverrideStat>>,
    bind_identity_map: Option<&BindIdentityMapHandle>,
) -> io::Result<stat64> {
    match result {
        Ok(Some(ovr)) => {
            st.st_uid = ovr.uid;
            st.st_gid = ovr.gid;

            #[cfg(target_os = "linux")]
            {
                st.st_mode = ovr.mode;
            }
            #[cfg(target_os = "macos")]
            {
                st.st_mode = ovr.mode as u16;
            }

            if ovr.mode & platform::MODE_TYPE_MASK == platform::MODE_BLK
                || ovr.mode & platform::MODE_TYPE_MASK == platform::MODE_CHR
            {
                #[cfg(target_os = "linux")]
                {
                    st.st_rdev = u64::from(ovr.rdev);
                }
                #[cfg(target_os = "macos")]
                {
                    st.st_rdev = ovr.rdev as i32;
                }
            }
            Ok(st)
        }
        Ok(None) => {
            apply_bind_identity_map(&mut st, bind_identity_map);
            Ok(st)
        } // No override xattr
        Err(e) => Err(e),
    }
}

/// Read the override xattr from a file descriptor.
///
/// Returns `None` if the xattr does not exist (ENODATA/ENOATTR).
/// Returns `Err(EIO)` for corrupt data.
fn read_override(fd: RawFd, strict: bool) -> io::Result<Option<OverrideStat>> {
    let mut buf = [0u8; OVERRIDE_SIZE];

    #[cfg(target_os = "linux")]
    let mut path_buf = [0u8; PROC_SELF_FD_PATH_BUF_LEN];

    #[cfg(target_os = "linux")]
    let path = format_proc_self_fd_path(fd, &mut path_buf)?;

    #[cfg(target_os = "linux")]
    let ret = unsafe {
        libc::getxattr(
            path,
            OVERRIDE_XATTR_KEY.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            OVERRIDE_SIZE,
        )
    };

    #[cfg(target_os = "macos")]
    let ret = unsafe {
        libc::fgetxattr(
            fd,
            OVERRIDE_XATTR_KEY.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            OVERRIDE_SIZE,
            0,
            0,
        )
    };

    parse_override(ret, &buf, strict)
}

/// Read the override xattr from `path` itself, not from what a final symlink points to.
#[cfg(target_os = "macos")]
fn read_override_at_path(path: &CStr, strict: bool) -> io::Result<Option<OverrideStat>> {
    let mut buf = [0u8; OVERRIDE_SIZE];
    let ret = unsafe {
        libc::getxattr(
            path.as_ptr(),
            OVERRIDE_XATTR_KEY.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            OVERRIDE_SIZE,
            0,
            libc::XATTR_NOFOLLOW,
        )
    };
    parse_override(ret, &buf, strict)
}

fn parse_override(
    ret: isize,
    buf: &[u8; OVERRIDE_SIZE],
    strict: bool,
) -> io::Result<Option<OverrideStat>> {
    if ret < 0 {
        let err = io::Error::last_os_error();
        let errno = err.raw_os_error().unwrap_or(0);

        // ENODATA (Linux) or ENOATTR (macOS) means no xattr set.
        #[cfg(target_os = "linux")]
        if errno == libc::ENODATA {
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        if errno == libc::ENOATTR {
            return Ok(None);
        }

        // EOPNOTSUPP / ENOTSUP means the filesystem doesn't support xattrs.
        if errno == libc::EOPNOTSUPP || errno == libc::ENOTSUP {
            return handle_unsupported_xattr(strict);
        }

        return Err(platform::linux_error(err));
    }

    let size = ret as usize;
    if size < OVERRIDE_SIZE {
        return Err(platform::eio());
    }

    // SAFETY: buf is fully initialized and OVERRIDE_SIZE bytes long.
    let ovr: OverrideStat =
        unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const OverrideStat) };

    if ovr.version != OVERRIDE_VERSION {
        return Err(platform::eio());
    }

    Ok(Some(ovr))
}

fn handle_unsupported_xattr(strict: bool) -> io::Result<Option<OverrideStat>> {
    if strict {
        return Err(platform::eio());
    }

    Ok(None)
}

#[cfg(target_os = "linux")]
fn format_proc_self_fd_path(
    fd: RawFd,
    buf: &mut [u8; PROC_SELF_FD_PATH_BUF_LEN],
) -> io::Result<*const libc::c_char> {
    if fd < 0 {
        return Err(platform::eio());
    }

    let prefix = b"/proc/self/fd/";
    buf[..prefix.len()].copy_from_slice(prefix);

    let mut digits = [0u8; 10];
    let mut value = fd as u32;
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }

    let digits = &digits[start..];
    let end = prefix.len() + digits.len();
    if end + 1 > buf.len() {
        return Err(platform::eio());
    }

    buf[prefix.len()..end].copy_from_slice(digits);
    buf[end] = 0;
    Ok(buf.as_ptr().cast())
}

pub(crate) fn apply_bind_identity_map(
    st: &mut stat64,
    bind_identity_map: Option<&BindIdentityMapHandle>,
) {
    if let Some(map) = bind_identity_map.and_then(|handle| handle.get().copied()) {
        map.apply(st);
    }
}

fn is_mapped(bind_identity_map: Option<&BindIdentityMapHandle>) -> bool {
    bind_identity_map.is_some_and(|handle| handle.get().is_some())
}

fn ctime(st: &stat64) -> (i64, i64) {
    (st.st_ctime, st.st_ctime_nsec)
}

/// Set the override xattr on a file descriptor.
pub(crate) fn set_override(fd: RawFd, uid: u32, gid: u32, mode: u32, rdev: u32) -> io::Result<()> {
    let ovr = OverrideStat {
        version: OVERRIDE_VERSION,
        _pad: [0; 3],
        uid,
        gid,
        mode,
        rdev,
    };

    let buf = unsafe {
        std::slice::from_raw_parts(&ovr as *const OverrideStat as *const u8, OVERRIDE_SIZE)
    };

    #[cfg(target_os = "linux")]
    {
        let mut path_buf = [0u8; PROC_SELF_FD_PATH_BUF_LEN];
        let path = format_proc_self_fd_path(fd, &mut path_buf)?;
        let ret = unsafe {
            libc::setxattr(
                path,
                OVERRIDE_XATTR_KEY.as_ptr(),
                buf.as_ptr() as *const libc::c_void,
                OVERRIDE_SIZE,
                0,
            )
        };
        if ret < 0 {
            return Err(platform::linux_error(io::Error::last_os_error()));
        }
    }

    #[cfg(target_os = "macos")]
    {
        let ret = unsafe {
            libc::fsetxattr(
                fd,
                OVERRIDE_XATTR_KEY.as_ptr(),
                buf.as_ptr() as *const libc::c_void,
                OVERRIDE_SIZE,
                0,
                0,
            )
        };
        if ret < 0 {
            return Err(platform::linux_error(io::Error::last_os_error()));
        }
    }

    Ok(())
}

/// Set the override xattr on a path (for use when we don't have an fd).
pub(crate) fn set_override_at(
    dirfd: RawFd,
    name: &CStr,
    uid: u32,
    gid: u32,
    mode: u32,
    rdev: u32,
) -> io::Result<()> {
    // Open the file to get an fd, then delegate.
    let fd = unsafe {
        libc::openat(
            dirfd,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        // For directories, use O_RDONLY | O_DIRECTORY.
        let fd = unsafe {
            libc::openat(
                dirfd,
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_DIRECTORY,
            )
        };
        if fd < 0 {
            return Err(platform::linux_error(io::Error::last_os_error()));
        }
        let result = set_override(fd, uid, gid, mode, rdev);
        unsafe { libc::close(fd) };
        return result;
    }
    let result = set_override(fd, uid, gid, mode, rdev);
    unsafe { libc::close(fd) };
    result
}

/// Read the current override xattr values from a file descriptor.
///
/// Returns `None` if no override is set.
pub(crate) fn get_override(
    fd: RawFd,
    xattr_enabled: bool,
    strict: bool,
) -> io::Result<Option<OverrideStat>> {
    if !xattr_enabled {
        return Ok(None);
    }

    read_override(fd, strict)
}

/// Check whether xattrs can be read from the given directory without mutating it.
///
/// An absent override still proves that the lookup reached an xattr-capable
/// filesystem. Read-only passthrough mounts use this probe because they never
/// need to persist guest metadata changes.
///
/// Returns `Ok(true)` if xattr reads work, `Ok(false)` if they are unsupported.
pub(crate) fn probe_xattr_read_support(dirfd: RawFd) -> io::Result<bool> {
    #[cfg(target_os = "linux")]
    let ret = {
        let mut path_buf = [0u8; PROC_SELF_FD_PATH_BUF_LEN];
        let path = format_proc_self_fd_path(dirfd, &mut path_buf)?;
        unsafe { libc::getxattr(path, OVERRIDE_XATTR_KEY.as_ptr(), std::ptr::null_mut(), 0) }
    };

    #[cfg(target_os = "macos")]
    let ret = unsafe {
        libc::fgetxattr(
            dirfd,
            OVERRIDE_XATTR_KEY.as_ptr(),
            std::ptr::null_mut(),
            0,
            0,
            0,
        )
    };

    if ret >= 0 {
        return Ok(true);
    }

    let err = io::Error::last_os_error();
    let errno = err.raw_os_error().unwrap_or(0);

    #[cfg(target_os = "linux")]
    if errno == libc::ENODATA {
        return Ok(true);
    }
    #[cfg(target_os = "macos")]
    if errno == libc::ENOATTR {
        return Ok(true);
    }

    if errno == libc::EOPNOTSUPP || errno == libc::ENOTSUP {
        return Ok(false);
    }

    Err(platform::linux_error(err))
}

/// Check if xattrs can be written by probing the given directory.
///
/// Returns `Ok(true)` if xattr writes work, `Ok(false)` if they are unsupported.
pub(crate) fn probe_xattr_write_support(dirfd: RawFd) -> io::Result<bool> {
    let probe_val: [u8; 1] = [1];

    #[cfg(target_os = "linux")]
    {
        let mut path_buf = [0u8; PROC_SELF_FD_PATH_BUF_LEN];
        let path = format_proc_self_fd_path(dirfd, &mut path_buf)?;
        let ret = unsafe {
            libc::setxattr(
                path,
                PROBE_XATTR_KEY.as_ptr(),
                probe_val.as_ptr() as *const libc::c_void,
                1,
                0,
            )
        };
        if ret < 0 {
            let err = io::Error::last_os_error();
            let errno = err.raw_os_error().unwrap_or(0);
            if errno == libc::EOPNOTSUPP || errno == libc::ENOTSUP {
                return Ok(false);
            }
            return Err(platform::linux_error(err));
        }
        // Clean up the probe xattr.
        unsafe {
            libc::removexattr(path, PROBE_XATTR_KEY.as_ptr());
        }
    }

    #[cfg(target_os = "macos")]
    {
        let ret = unsafe {
            libc::fsetxattr(
                dirfd,
                PROBE_XATTR_KEY.as_ptr(),
                probe_val.as_ptr() as *const libc::c_void,
                1,
                0,
                0,
            )
        };
        if ret < 0 {
            let err = io::Error::last_os_error();
            let errno = err.raw_os_error().unwrap_or(0);
            if errno == libc::EOPNOTSUPP || errno == libc::ENOTSUP {
                return Ok(false);
            }
            return Err(platform::linux_error(err));
        }
        // Clean up the probe xattr.
        unsafe {
            libc::fremovexattr(dirfd, PROBE_XATTR_KEY.as_ptr(), 0);
        }
    }

    Ok(true)
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{OVERRIDE_XATTR_KEY, PROBE_XATTR_KEY, handle_unsupported_xattr};

    fn host_stat(ctime: i64) -> crate::stat64 {
        let mut st: crate::stat64 = unsafe { std::mem::zeroed() };
        st.st_ctime = ctime as _;
        st.st_uid = 501;
        st.st_gid = 20;
        st.st_mode = (libc::S_IFREG | 0o644) as _;
        st.st_size = 10;
        st
    }

    fn patch(host: &crate::stat64) -> crate::stat64 {
        let mut st = *host;
        st.st_uid = 0;
        st.st_gid = 0;
        st.st_mode = (libc::S_IFREG | 0o600) as _;
        st
    }

    #[test]
    #[allow(clippy::unnecessary_cast)]
    fn test_patched_stat_cache_reuses_patch_for_same_ctime() {
        let cache = super::PatchedStatCache::default();
        let host = host_stat(100);
        cache.store(&host, &patch(&host), None);

        let mut later = host;
        later.st_size = 20;
        later.st_mtime = 5;
        let st = cache
            .get(&later, None)
            .expect("same ctime reuses the patch");
        assert_eq!(
            (st.st_uid, st.st_gid, st.st_mode as u32),
            (0, 0, libc::S_IFREG as u32 | 0o600)
        );
        assert_eq!(
            st.st_size, 20,
            "fields the override doesn't replace come from the new stat"
        );
    }

    #[test]
    fn test_patched_stat_cache_misses_on_changed_source() {
        let cache = super::PatchedStatCache::default();
        assert!(cache.get(&host_stat(100), None).is_none());

        let host = host_stat(100);
        cache.store(&host, &patch(&host), None);
        assert!(cache.get(&host_stat(101), None).is_none());
        let mut nsec = host;
        nsec.st_ctime_nsec = 1;
        assert!(cache.get(&nsec, None).is_none());
        let mut owner = host;
        owner.st_uid = 502;
        assert!(cache.get(&owner, None).is_none());
    }

    #[test]
    fn test_patched_stat_cache_misses_after_identity_map_installed() {
        let handle: super::BindIdentityMapHandle = Default::default();
        let cache = super::PatchedStatCache::default();
        let host = host_stat(100);
        cache.store(&host, &host, Some(&handle));
        assert!(cache.get(&host, Some(&handle)).is_some());

        handle
            .set(super::BindIdentityMap::new(501, 1000, 1000))
            .unwrap();
        assert!(cache.get(&host, Some(&handle)).is_none());
    }

    #[test]
    fn test_xattr_keys_use_msb_namespace() {
        assert_eq!(OVERRIDE_XATTR_KEY.to_bytes(), b"user.msb.override_stat");
        assert_eq!(PROBE_XATTR_KEY.to_bytes(), b"user.msb._probe");
    }

    #[test]
    fn test_unsupported_xattr_is_eio_in_strict_mode() {
        let err = match handle_unsupported_xattr(true) {
            Ok(_) => panic!("strict mode must hard-fail on unsupported xattrs"),
            Err(err) => err,
        };
        assert_eq!(err.raw_os_error(), Some(libc::EIO));
    }

    #[test]
    fn test_unsupported_xattr_is_none_in_non_strict_mode() {
        assert!(handle_unsupported_xattr(false).unwrap().is_none());
    }

    #[test]
    fn test_bind_identity_map_applies_owner_and_overflow() {
        use super::BindIdentityMap;

        let map = BindIdentityMap::new(501, 1000, 1000);

        // A file owned by the host owner is presented as the mapped guest owner.
        let mut st: crate::stat64 = unsafe { std::mem::zeroed() };
        st.st_uid = 501;
        st.st_gid = 20;
        map.apply(&mut st);
        assert_eq!(st.st_uid, 1000);
        assert_eq!(st.st_gid, 1000);

        // Any other owner falls through to the overflow identity (65534).
        let mut other: crate::stat64 = unsafe { std::mem::zeroed() };
        other.st_uid = 4242;
        other.st_gid = 4242;
        map.apply(&mut other);
        assert_eq!(other.st_uid, 65534);
        assert_eq!(other.st_gid, 65534);
    }

    #[test]
    fn test_fixed_bind_identity_map_applies_to_every_host_owner() {
        use super::BindIdentityMap;

        let map = BindIdentityMap::fixed(1000, 1001);
        for host_uid in [0, 501, 4242, u32::MAX] {
            let mut st: crate::stat64 = unsafe { std::mem::zeroed() };
            st.st_uid = host_uid;
            st.st_gid = host_uid;
            map.apply(&mut st);
            assert_eq!((st.st_uid, st.st_gid), (1000, 1001));
        }
    }
}
