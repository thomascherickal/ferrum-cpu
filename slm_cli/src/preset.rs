//! `train --preset tiny|small|medium`: the starting sizes from Manual §6.7 as defaults.
//! An explicit flag always wins; without `--preset` nothing changes.

use crate::args::Args;
use ferrum_core::TransformerConfig;

pub(crate) struct Preset {
    pub(crate) name: &'static str,
    context: usize,
    embed: usize,
    heads: usize,
    blocks: usize,
    hidden: usize,
    vocab: usize,
    lr: f32,
}

pub(crate) const PRESETS: [Preset; 3] = [
    // ~0.2M parameters, minutes.
    Preset {
        name: "tiny",
        context: 64,
        embed: 64,
        heads: 4,
        blocks: 2,
        hidden: 256,
        vocab: 512,
        lr: 1e-2,
    },
    // ~1.5M parameters, hours.
    Preset {
        name: "small",
        context: 128,
        embed: 128,
        heads: 4,
        blocks: 4,
        hidden: 512,
        vocab: 2048,
        lr: 3e-3,
    },
    // ~7M parameters, a day or two.
    Preset {
        name: "medium",
        context: 256,
        embed: 256,
        heads: 8,
        blocks: 6,
        hidden: 1024,
        vocab: 4096,
        lr: 1e-3,
    },
];

/// The preset named by `--preset`, `Ok(None)` when the flag is absent.
pub(crate) fn selected(args: &Args) -> Result<Option<&'static Preset>, String> {
    let Some(name) = args.value("preset") else {
        return Ok(None);
    };
    PRESETS
        .iter()
        .find(|p| p.name == name)
        .map(Some)
        .ok_or_else(|| {
            let valid: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
            format!("unknown --preset '{name}' (valid: {})", valid.join(", "))
        })
}

impl Preset {
    /// `base` with this preset's sizes in place of its defaults.
    pub(crate) fn defaults(&self, base: TransformerConfig) -> TransformerConfig {
        TransformerConfig {
            context_len: self.context,
            embed_dim: self.embed,
            num_heads: self.heads,
            num_blocks: self.blocks,
            hidden_dim: self.hidden,
            vocab_size: self.vocab,
            lr: self.lr,
            ..base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::spec_for;

    fn args(a: &[&str]) -> Args {
        let v: Vec<String> = a.iter().map(|s| s.to_string()).collect();
        Args::parse(&v, spec_for("train").unwrap()).unwrap()
    }

    #[test]
    fn no_preset_leaves_the_defaults() {
        let a = args(&["--epochs", "3"]);
        assert!(selected(&a).unwrap().is_none());
        let d = TransformerConfig::default();
        let c = a.config();
        assert_eq!(
            (c.context_len, c.embed_dim, c.lr),
            (d.context_len, d.embed_dim, d.lr)
        );
    }

    #[test]
    fn each_preset_sets_the_manual_sizes() {
        let c = args(&["--preset", "small"]).config();
        assert_eq!(
            (
                c.context_len,
                c.embed_dim,
                c.num_heads,
                c.num_blocks,
                c.hidden_dim,
                c.vocab_size
            ),
            (128, 128, 4, 4, 512, 2048)
        );
        assert_eq!(c.lr, 3e-3);
        let c = args(&["--preset", "medium"]).config();
        assert_eq!(
            (c.num_blocks, c.hidden_dim, c.vocab_size, c.lr),
            (6, 1024, 4096, 1e-3)
        );
    }

    #[test]
    fn explicit_flags_override_the_preset_in_any_order() {
        for raw in [
            ["--preset", "tiny", "--embed", "32", "--lr", "0.5"],
            ["--embed", "32", "--lr", "0.5", "--preset", "tiny"],
        ] {
            let c = args(&raw).config();
            assert_eq!((c.embed_dim, c.lr), (32, 0.5), "{raw:?}");
            assert_eq!((c.context_len, c.hidden_dim, c.vocab_size), (64, 256, 512));
        }
    }

    #[test]
    fn unknown_preset_lists_the_valid_ones() {
        let e = selected(&args(&["--preset", "huge"])).err().unwrap();
        assert_eq!(e, "unknown --preset 'huge' (valid: tiny, small, medium)");
    }
}
