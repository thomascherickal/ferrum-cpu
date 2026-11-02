//! `train --resume`: the checkpoint's architecture wins over the flags, so say so.

use crate::args::*;
use ferrum_core::{CheckpointInfo, TransformerConfig, TransformerNet};

/// Header of the checkpoint a resumed run will continue from (`None` if unreadable; the trainer
/// then reports the real error itself).
pub(crate) fn read_info(path: &str) -> Option<CheckpointInfo> {
    TransformerNet::checkpoint_info(std::fs::read(path).ok()?).ok()
}

/// One `note:` line per flag the checkpoint overrides. (The corpus needs no note: the trainer
/// refuses a resume on a different one.)
pub(crate) fn notes(args: &Args, info: &CheckpointInfo) -> Vec<String> {
    let mut out = Vec::new();
    for (flag, used) in [
        ("embed", info.embed),
        ("heads", info.heads),
        ("hidden", info.hidden),
        ("blocks", info.blocks),
    ] {
        if let Some(v) = args.value(flag) {
            if v.parse::<usize>().ok() != Some(used) {
                out.push(format!(
                    "note: --{flag} {v} ignored, using {used} from the checkpoint"
                ));
            }
        }
    }
    if let Some(v) = args.value("seed") {
        out.push(format!(
            "note: --seed {v} ignored, the checkpoint's random state is used"
        ));
    }
    out
}

/// Print the notes and make `cfg` show what will really be used.
pub(crate) fn apply(args: &Args, cfg: &mut TransformerConfig, info: &CheckpointInfo) {
    for n in notes(args, info) {
        println!("{n}");
    }
    cfg.embed_dim = info.embed;
    cfg.num_heads = info.heads;
    cfg.hidden_dim = info.hidden;
    cfg.num_blocks = info.blocks;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> CheckpointInfo {
        CheckpointInfo {
            vocab: 300,
            context: 16,
            embed: 32,
            heads: 4,
            hidden: 64,
            blocks: 2,
            step: 10,
        }
    }

    fn args(a: &[&str]) -> Args {
        let v: Vec<String> = a.iter().map(|s| s.to_string()).collect();
        Args::parse(&v, spec_for("train").unwrap()).unwrap()
    }

    #[test]
    fn conflicting_flags_get_one_note_each_and_matching_ones_none() {
        let n = notes(
            &args(&[
                "--embed", "64", "--heads", "4", "--blocks", "3", "--seed", "7",
            ]),
            &info(),
        );
        assert_eq!(
            n[0],
            "note: --embed 64 ignored, using 32 from the checkpoint"
        );
        assert_eq!(
            n[1],
            "note: --blocks 3 ignored, using 2 from the checkpoint"
        );
        assert!(n[2].starts_with("note: --seed 7 ignored"), "{n:?}");
        assert_eq!(n.len(), 3, "{n:?}");
        assert!(notes(&args(&[]), &info()).is_empty());
    }

    #[test]
    fn apply_overwrites_the_architecture_shown_in_the_banner() {
        let a = args(&["--embed", "64"]);
        let mut cfg = a.config();
        apply(&a, &mut cfg, &info());
        assert_eq!(
            (cfg.embed_dim, cfg.num_heads, cfg.hidden_dim, cfg.num_blocks),
            (32, 4, 64, 2)
        );
    }
}
