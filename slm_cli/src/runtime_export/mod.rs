//! `export-ollama` / `export-lmstudio`: put a Ferrum GGUF into a local runtime
//! with one command. Only `gpt2`-architecture GGUF files run in llama.cpp,
//! so anything else is refused before any tool is touched.

mod find;
mod lmstudio;
mod ollama;

use std::path::PathBuf;

pub use find::{find_tool, plain_gguf_warning, runnable_summary, GgufSummary};
pub use lmstudio::{
    cmd_export_lmstudio, default_lmstudio_repo, export_lmstudio, lmstudio_preview, lmstudio_run,
    LmsRun,
};
pub use ollama::{
    cmd_export_ollama, export_ollama, modelfile_for, ollama_modelfile, ollama_preview, ollama_run,
    write_modelfile, OllamaRun,
};

/// The message of an export ended by the caller's cancel check.
pub const CANCELLED: &str = "export cancelled; nothing was run";

/// What an export would do, without doing it (nothing is run or contacted).
#[derive(Debug)]
pub struct Preview {
    /// The Modelfile text (Ollama only).
    pub modelfile: Option<String>,
    /// The tool that would run, or why it cannot be found.
    pub tool: Result<PathBuf, String>,
    /// The commands that would run, in order.
    pub commands: Vec<String>,
}

#[cfg(test)]
pub(super) mod test_support {
    use super::GgufSummary;
    use ferrum_core::{GgufBuilder, MetaValue};
    use std::ffi::OsStr;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};

    /// A fresh temp dir that is removed on drop (also when the test panics).
    pub struct Tmp(pub PathBuf);
    impl std::ops::Deref for Tmp {
        type Target = PathBuf;
        fn deref(&self) -> &PathBuf {
            &self.0
        }
    }
    impl AsRef<OsStr> for Tmp {
        fn as_ref(&self) -> &OsStr {
            self.0.as_os_str()
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub fn tmp(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ferrum_rx_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    pub fn gguf(dir: &Path, file: &str, arch: Option<&str>, ctx: Option<u32>) -> PathBuf {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.meta("general.architecture", MetaValue::String(a.into()));
        }
        if let Some(c) = ctx {
            b.meta("gpt2.context_length", MetaValue::U32(c));
        }
        let p = dir.join(file);
        std::fs::write(&p, b.into_bytes()).unwrap();
        p
    }

    /// An executable shell script named `name` in `dir`.
    #[cfg(unix)]
    pub fn fake(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    pub fn s(p: &Path) -> &str {
        p.to_str().unwrap()
    }

    pub fn summary(ctx: u32) -> GgufSummary {
        GgufSummary { context_len: ctx }
    }

    pub fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|x| x.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_exports_expose_shared_constants_and_types() {
        assert_eq!(CANCELLED, "export cancelled; nothing was run");
        let prev = Preview {
            modelfile: Some("FROM \"m.gguf\"\n".into()),
            tool: Ok(PathBuf::from("/bin/ollama")),
            commands: vec!["ollama create m".into()],
        };
        assert!(format!("{prev:?}").contains("ollama create m"));
    }
}
