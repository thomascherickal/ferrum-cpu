//! Training hyperparameters and validation-split controls.
use crate::slm::scoring::Evaluation;
use crate::stop::StopCondition;

/// Hyperparameters for training a causal transformer SLM.
///
/// Used by `GenerativeSLM::train_transformer_config` and
/// `GenerativeSLM::load_or_train`. `Default` matches the CLI defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformerConfig {
    /// Context window in characters.
    pub context_len: usize,
    /// Embedding dimension (must be divisible by `num_heads`).
    pub embed_dim: usize,
    /// Attention heads per block.
    pub num_heads: usize,
    /// Number of transformer blocks.
    pub num_blocks: usize,
    /// FFN hidden width.
    pub hidden_dim: usize,
    /// Training epochs.
    pub epochs: usize,
    /// Adam learning rate.
    pub lr: f32,
    /// Minibatch size (sequences per step).
    pub batch_size: usize,
    /// Byte-level BPE target vocabulary size. `0` selects character-level
    /// tokenization (legacy behaviour); any value `>= 256` trains a BPE
    /// tokenizer of that size and stores it inside the model.
    pub vocab_size: usize,
    /// Decoupled (AdamW) weight-decay coefficient applied to weight matrices
    /// (T7). `0.0` disables it; typical values are 0.01–0.1.
    pub weight_decay: f32,
    /// FFN-hidden dropout probability used during training (T7), in `[0, 1)`.
    /// `0.0` disables it; inference is always dropout-free.
    pub dropout: f32,
    /// Tokens between training-window starts. `0` = `context_len`
    /// (non-overlapping: one epoch sees each token once). `1` = the legacy
    /// fully-overlapping window, where an epoch costs ≈`context_len`× more.
    pub window_stride: usize,
    /// Training budget in tokens (`steps × batch_size × context_len`). `0`
    /// trains for `epochs`; otherwise `epochs` is ignored and training stops
    /// once the budget is spent, mid-epoch if need be.
    pub max_tokens: u64,
    /// Int8 quantization-aware training.
    pub qat: bool,
    /// Global-norm gradient clipping threshold; `0.0` disables it.
    pub grad_clip: f32,
    /// Linear warmup then cosine decay to 0 over the whole run (`false` =
    /// fixed `lr`).
    pub cosine_lr: bool,
    /// Warmup steps for `cosine_lr`.
    pub warmup_steps: u64,
    /// Tie the LM head to the token embedding (saves `vocab × embed` params).
    pub weight_tying: bool,
    /// BPE tokenizer file: loaded when it exists, otherwise trained and
    /// written there, so later runs skip tokenizer training. Empty = train in
    /// memory every run. Ignored for character-level (`vocab_size == 0`).
    pub tokenizer_path: String,
    /// Training-checkpoint file (weights + Adam state + RNG). Empty disables.
    pub checkpoint_path: String,
    /// Checkpoint every this many optimizer steps (and always at the end);
    /// `0` = only at the end.
    pub checkpoint_every: u64,
    /// Resume from `checkpoint_path` when it exists. The checkpoint fixes the
    /// architecture; its vocab and context must match this run's.
    pub resume: bool,
    /// Print a `[train] step S/T  tokens/s R  elapsed Xs` line every this many
    /// optimizer steps; `0` = off.
    pub log_every_steps: u64,
    /// Write one CSV row per optimizer step (`step,epoch,loss,lr,grad_norm,tokens_per_sec,
    /// elapsed`) to this file; empty = off. A resumed run appends. Turning it on measures
    /// the gradient norm each step, which does not change the trained weights.
    pub metrics_csv: String,
    /// Graceful stop (deadline, stop file, cancel flag), polled after every
    /// optimizer step. On a stop the run checkpoints (if `checkpoint_path` is set)
    /// and returns the model as trained so far; [`StopCondition::stopped`] says
    /// where. The default never stops.
    pub stop: StopCondition,
    /// Skip the up-front memory refusal (training peaks above 16 GB or above three quarters
    /// of the free RAM). For a caller that has shown the user the estimate and been told to go
    /// on anyway (the GUI's "I know, try anyway"); the operating system may kill the process
    /// if the run really does not fit. The CLI never sets it. Default `false`.
    pub skip_memory_check: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            context_len: 16,
            embed_dim: 32,
            num_heads: 4,
            num_blocks: 2,
            hidden_dim: 64,
            epochs: 100,
            lr: 0.01,
            batch_size: 16,
            vocab_size: 512,
            weight_decay: 0.0,
            dropout: 0.0,
            window_stride: 0,
            max_tokens: 0,
            qat: true,
            grad_clip: 0.0,
            cosine_lr: false,
            warmup_steps: 0,
            weight_tying: false,
            tokenizer_path: String::new(),
            checkpoint_path: String::new(),
            checkpoint_every: 0,
            resume: false,
            log_every_steps: 0,
            metrics_csv: String::new(),
            stop: StopCondition::default(),
            skip_memory_check: false,
        }
    }
}

/// Validation-aware training controls (T5): an internal held-out split, early
/// stopping on validation perplexity, and best-epoch checkpointing.
#[derive(Clone, Debug)]
pub struct ValidationConfig {
    /// Fraction of the corpus (in `0.0..1.0`) held out at the **tail** for
    /// validation. The tokenizer/vocabulary is fit on the training portion only.
    pub val_fraction: f32,
    /// Early stopping: stop once validation cross-entropy fails to improve for
    /// this many consecutive epochs. `0` disables early stopping (all epochs run)
    /// but the best-by-validation checkpoint is still returned.
    pub patience: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            val_fraction: 0.1,
            patience: 0,
        }
    }
}

/// Per-epoch report from validation-aware training (T5).
#[derive(Clone, Debug)]
pub struct ValidationProgress {
    pub epoch: usize,
    pub train_loss: f32,
    /// Held-out evaluation for this epoch.
    pub val: Evaluation,
    /// Whether this epoch set a new best validation cross-entropy (and is the
    /// checkpoint that would be returned if training stopped now).
    pub is_best: bool,
}

#[cfg(test)]
mod tests {

    use super::{TransformerConfig, ValidationConfig};

    #[test]
    fn validation_defaults_hold_out_a_tenth_without_early_stopping() {
        let v = ValidationConfig::default();
        assert_eq!(v.val_fraction, 0.1);
        assert_eq!(v.patience, 0);
        // The training defaults the CLI documents.
        let c = TransformerConfig::default();
        assert!(c.context_len > 0 && c.embed_dim > 0 && c.epochs > 0);
    }
}
