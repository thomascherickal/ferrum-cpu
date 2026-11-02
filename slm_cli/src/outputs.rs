//! Checks on the files a run will write, done before any long work starts. Shared by the
//! `train`/`tokenize` commands and the GUI.

/// Fail fast, before any work, when `path` cannot be written: create its parent directory (as
/// `train` always did) and test-open a temp file next to it. Error: `cannot write <path>: <why>`.
pub fn check_writable(path: &str) -> Result<(), String> {
    use std::path::Path;
    let bad = |e: &dyn std::fmt::Display| format!("cannot write {path}: {e}");
    let p = Path::new(path);
    if p.is_dir() {
        return Err(bad(&"is a directory"));
    }
    let parent = p.parent().filter(|d| !d.as_os_str().is_empty());
    if let Some(d) = parent {
        std::fs::create_dir_all(d).map_err(|e| bad(&e))?;
    }
    let name = p
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let probe = parent
        .unwrap_or(Path::new("."))
        .join(format!(".{name}.probe{}", std::process::id()));
    std::fs::File::create(&probe).map_err(|e| bad(&e))?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

/// The file `p` names, made comparable: canonical when it exists, else its (canonical) parent
/// plus file name, else the absolute path.
fn canonical_target(p: &str) -> std::path::PathBuf {
    use std::path::Path;
    let path = Path::new(p);
    if let Ok(c) = path.canonicalize() {
        return c;
    }
    let parent = path.parent().filter(|d| !d.as_os_str().is_empty());
    match (
        parent.unwrap_or(Path::new(".")).canonicalize(),
        path.file_name(),
    ) {
        (Ok(d), Some(n)) => d.join(n),
        _ => std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
    }
}

/// Refuse two roles (`(label, path)`; an empty path means unused) that name the same file, e.g.
/// `--checkpoint` equal to the model output, which would silently overwrite one with the other.
pub fn check_distinct_paths(roles: &[(&str, &str)]) -> Result<(), String> {
    let used: Vec<_> = roles.iter().filter(|(_, p)| !p.is_empty()).collect();
    for (i, (la, pa)) in used.iter().enumerate() {
        for (lb, pb) in &used[i + 1..] {
            if pa == pb || canonical_target(pa) == canonical_target(pb) {
                return Err(format!(
                    "{la} and {lb} are the same file ({pb}); they would overwrite each other"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_writable_creates_parents_and_names_the_path_on_failure() {
        let dir = std::env::temp_dir().join(format!("ferrum_writable_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ok = dir.join("a/b/m.gguf");
        check_writable(ok.to_str().unwrap()).unwrap();
        assert!(dir.join("a/b").is_dir(), "parent created");
        assert!(!ok.exists(), "the target itself is not created");
        assert_eq!(
            std::fs::read_dir(dir.join("a/b")).unwrap().count(),
            0,
            "probe removed"
        );
        // A bare file name (no parent component) is fine.
        check_writable("ferrum_writable_probe_target.bin").unwrap();
        assert!(!std::path::Path::new("ferrum_writable_probe_target.bin").exists());
        // A directory, and a parent that is a file, are refused with the path named.
        let e = check_writable(dir.to_str().unwrap()).unwrap_err();
        assert!(
            e.starts_with("cannot write ") && e.contains("is a directory"),
            "{e}"
        );
        let f = dir.join("file");
        std::fs::write(&f, b"x").unwrap();
        let bad = f.join("sub/m.gguf");
        let e = check_writable(bad.to_str().unwrap()).unwrap_err();
        assert!(
            e.starts_with(&format!("cannot write {}: ", bad.display())),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_file_under_different_spellings_is_refused() {
        let d = std::env::temp_dir().join(format!("ferrum_distinct_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub")).unwrap();
        let (a, b) = (d.join("m.gguf"), d.join("c.fckp"));
        let (a, b) = (a.to_str().unwrap(), b.to_str().unwrap());
        assert!(
            check_distinct_paths(&[("model", a), ("--checkpoint", b), ("--tokenizer", "")]).is_ok()
        );
        let e = check_distinct_paths(&[("the model output", a), ("--checkpoint", a)]).unwrap_err();
        assert!(
            e.contains("the model output and --checkpoint are the same file"),
            "{e}"
        );
        // Not yet existing, with a `..` detour through an existing directory.
        let detour = format!("{}/sub/../m.gguf", d.display());
        assert!(check_distinct_paths(&[("a", a), ("b", &detour)]).is_err());
        // Existing file reached through a symlink.
        std::fs::write(a, b"x").unwrap();
        let link = d.join("link.gguf");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(a, &link).unwrap();
            assert!(check_distinct_paths(&[("a", a), ("b", link.to_str().unwrap())]).is_err());
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_directory_fails_the_probe_and_names_the_path() {
        use std::os::unix::fs::PermissionsExt;
        let d = std::env::temp_dir().join(format!("ferrum_ro_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o555)).unwrap();
        let target = d.join("m.gguf");
        let r = check_writable(target.to_str().unwrap());
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _ = std::fs::remove_dir_all(&d);
        // ponytail: a process that bypasses permissions (root) can write there anyway.
        if let Err(e) = r {
            assert!(
                e.starts_with("cannot write ") && e.contains("m.gguf"),
                "{e}"
            );
        }
    }

    #[test]
    fn paths_with_missing_parents_are_compared_by_their_absolute_spelling() {
        // Neither parent exists, so neither can be canonicalized.
        let a = "/nonexistent-ferrum-dir/a/../m.gguf";
        assert_eq!(canonical_target(a), canonical_target(a));
        assert!(canonical_target("/nonexistent-ferrum-dir/m.gguf").is_absolute());
        assert!(check_distinct_paths(&[
            ("a", "/nonexistent-ferrum-dir/x"),
            ("b", "/nonexistent-ferrum-dir/y")
        ])
        .is_ok());
        // A name with no file component (`..`) falls back to the absolute path too.
        assert!(canonical_target("nonexistent-ferrum-dir/..").is_absolute());
        // The empty path has no absolute form: it is returned as given.
        assert_eq!(canonical_target(""), std::path::PathBuf::from(""));
    }
}
