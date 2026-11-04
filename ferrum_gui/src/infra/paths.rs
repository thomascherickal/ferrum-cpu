//! Path validation for every filesystem path the webview sends to a command.
//!
//! The webview is untrusted (a compromised page can call any command with any
//! string), so each command that reads, writes or creates a path runs it
//! through [`checked_path`] / [`checked_dir`] first. Relative paths are kept:
//! the UI uses free-text inputs, so a typed relative path is legitimate (it
//! resolves against the app's working directory, exactly as before).
//!
//! The checks are: non-empty, no NUL byte, at most [`MAX_PATH_BYTES`], not under
//! `/dev`, `/proc` or `/sys` (after symlinks and `..` are resolved), and when it
//! exists, a regular file (or a directory for [`checked_dir`]), so FIFOs,
//! sockets and device nodes are refused. They are not a sandbox: ordinary
//! files anywhere the user can write stay reachable, and there is a small
//! check-then-use window.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const MAX_PATH_BYTES: usize = 4096;
const FORBIDDEN_ROOTS: [&str; 3] = ["/dev", "/proc", "/sys"];

/// Validate a *file* path. `must_exist` = true for inputs; false for outputs
/// (the file may be new, but if it exists it must be a regular file).
/// `label` names the field in the error message ("model path", ...).
pub fn checked_path(label: &str, p: &str, must_exist: bool) -> Result<PathBuf, String> {
    check(label, p, must_exist, Want::File)
}

/// Validate a *directory* path (a download folder). It may not exist yet.
pub fn checked_dir(label: &str, p: &str) -> Result<PathBuf, String> {
    check(label, p, false, Want::Dir)
}

/// Validate a corpus input: an existing regular file *or* directory.
pub fn checked_input(label: &str, p: &str) -> Result<PathBuf, String> {
    check(label, p, true, Want::Either)
}

#[derive(Clone, Copy, PartialEq)]
enum Want {
    File,
    Dir,
    Either,
}

fn check(label: &str, p: &str, must_exist: bool, want: Want) -> Result<PathBuf, String> {
    if p.trim().is_empty() {
        return Err(format!("{label}: please provide a path"));
    }
    if p.contains('\0') {
        return Err(format!("{label}: path contains a NUL byte"));
    }
    if p.len() > MAX_PATH_BYTES {
        return Err(format!(
            "{label}: path is longer than {MAX_PATH_BYTES} bytes"
        ));
    }
    let path = PathBuf::from(p);

    // Resolve the nearest existing ancestor so symlinks and `..` cannot hide a
    // forbidden root, whether or not the leaf exists yet.
    let anchor = path
        .ancestors()
        .find(|a| !a.as_os_str().is_empty() && a.exists())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let real = anchor
        .canonicalize()
        .map_err(|e| format!("{label}: cannot resolve {p}: {e}"))?;
    if FORBIDDEN_ROOTS.iter().any(|r| real.starts_with(r)) {
        return Err(format!(
            "{label}: {p} is a device or system path and is not allowed"
        ));
    }

    match std::fs::metadata(&path) {
        Ok(m) if want == Want::Dir && !m.is_dir() => {
            Err(format!("{label}: {p} is not a directory"))
        }
        Ok(m) if want == Want::File && m.is_dir() => {
            Err(format!("{label}: {p} is a directory, expected a file"))
        }
        Ok(m) if want != Want::Dir && !(m.is_file() || (want == Want::Either && m.is_dir())) => {
            Err(format!(
                "{label}: {p} is not a regular file (device, pipe or socket)"
            ))
        }
        Ok(_) => Ok(path),
        Err(e) if e.kind() == ErrorKind::NotFound && !must_exist => Ok(path),
        Err(e) if e.kind() == ErrorKind::NotFound => Err(format!("{label}: {p} does not exist")),
        Err(e) => Err(format!("{label}: cannot access {p}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("ferrum_gui_paths_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn rejects_empty_nul_and_too_long() {
        assert!(checked_path("p", "", false)
            .unwrap_err()
            .contains("provide a path"));
        assert!(checked_path("p", "   ", true).is_err());
        assert!(checked_path("p", "a\0b", false)
            .unwrap_err()
            .contains("NUL"));
        let long = format!("/tmp/{}", "a".repeat(MAX_PATH_BYTES));
        assert!(checked_path("p", &long, false)
            .unwrap_err()
            .contains("longer"));
        assert!(checked_dir("p", &long).is_err());
    }

    #[test]
    fn rejects_device_and_system_files() {
        for p in [
            "/dev/null",
            "/dev/zero",
            "/proc/self/mem",
            "/sys/kernel",
            "/dev/shm/x",
        ] {
            for must_exist in [true, false] {
                let e = checked_path("p", p, must_exist).unwrap_err();
                assert!(
                    e.contains("not allowed") || e.contains("does not exist"),
                    "{p}: {e}"
                );
            }
        }
        // `..` and symlinks cannot launder a forbidden root.
        assert!(checked_path("p", "/tmp/../dev/null", true)
            .unwrap_err()
            .contains("not allowed"));
        assert!(checked_dir("p", "/proc/new_dir")
            .unwrap_err()
            .contains("not allowed"));
        #[cfg(unix)]
        {
            let d = tmp("symlink");
            let link = d.join("zero");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink("/dev/zero", &link).unwrap();
            let e = checked_path("p", link.to_str().unwrap(), true).unwrap_err();
            assert!(e.contains("not allowed"), "{e}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_sockets_and_names_an_unreachable_path() {
        // A listening Unix socket is a file that is neither regular nor a directory.
        let d = tmp("socket");
        let sock = d.join("s");
        let _listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        let e = checked_path("p", sock.to_str().unwrap(), true).unwrap_err();
        assert!(e.contains("not a regular file"), "{e}");
        // A path below a regular file cannot be looked at: "not a directory" is not NotFound.
        let file = d.join("f");
        std::fs::write(&file, "x").unwrap();
        let below = file.join("child");
        let e = checked_path("p", below.to_str().unwrap(), true).unwrap_err();
        assert!(e.contains("cannot access"), "{e}");
    }

    #[test]
    fn directory_vs_file() {
        let d = tmp("dirfile");
        let ds = d.to_str().unwrap();
        assert!(checked_path("p", ds, true)
            .unwrap_err()
            .contains("is a directory"));
        assert!(checked_path("p", ds, false)
            .unwrap_err()
            .contains("is a directory"));
        assert!(checked_dir("p", ds).is_ok());
        let f = d.join("f.txt");
        std::fs::write(&f, "x").unwrap();
        assert!(checked_dir("p", f.to_str().unwrap())
            .unwrap_err()
            .contains("not a directory"));
    }

    #[test]
    fn input_may_be_a_file_or_a_folder_but_must_exist() {
        let d = tmp("input");
        let f = d.join("a.txt");
        std::fs::write(&f, "x").unwrap();
        assert!(checked_input("p", d.to_str().unwrap()).is_ok());
        assert!(checked_input("p", f.to_str().unwrap()).is_ok());
        assert!(checked_input("p", d.join("nope").to_str().unwrap())
            .unwrap_err()
            .contains("does not exist"));
        assert!(checked_input("p", "/dev/null")
            .unwrap_err()
            .contains("not allowed"));
        assert!(checked_input("p", "")
            .unwrap_err()
            .contains("provide a path"));
    }

    #[test]
    fn existence_rules() {
        let d = tmp("exist");
        let missing = d.join("missing.txt");
        let m = missing.to_str().unwrap();
        assert!(checked_path("p", m, true)
            .unwrap_err()
            .contains("does not exist"));
        // A new output path in an existing dir, and one whose parents are not yet created.
        assert_eq!(checked_path("p", m, false).unwrap(), missing);
        assert!(checked_path("p", d.join("a/b/c.txt").to_str().unwrap(), false).is_ok());
        // A new download folder is fine.
        assert!(checked_dir("p", d.join("new").to_str().unwrap()).is_ok());
    }

    #[test]
    fn accepts_normal_files_and_relative_paths() {
        let d = tmp("normal");
        let f = d.join("corpus.txt");
        std::fs::write(&f, "hello").unwrap();
        assert_eq!(checked_path("p", f.to_str().unwrap(), true).unwrap(), f);
        // Typed relative paths still work (resolved against the cwd).
        assert!(checked_path("p", "Cargo.toml", true).is_ok());
        assert!(checked_path("p", "out/new.gguf", false).is_ok());
    }

    #[test]
    fn error_names_the_field() {
        assert!(checked_path("model path", "", true)
            .unwrap_err()
            .starts_with("model path:"));
    }
}
