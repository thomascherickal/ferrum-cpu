//! Cross-checks of SLM metadata against the decoded layers.
use crate::error::Result;
use crate::layer::{Embedding, Linear, TransformerBlock};
use crate::loader::bad;
use crate::meta::{ModelMetadata, TaskType};
use crate::model::Sequential;
use crate::tokenizer::ByteBpeTokenizer;

/// A token-ID SLM's metadata (read from the untrusted file) must describe the
/// layers it ships with (checked by [`GenerativeSLM::from_bytes`](crate::slm::GenerativeSLM::from_bytes);
/// the codec itself stays a generic layer-list reader): generation and scoring size their buffers by
/// `input_dim`, index logits by `output_dim` and the char table by
/// `class_names`, so a file whose metadata disagrees with its weights would
/// otherwise panic or over-allocate there instead of failing to load.
pub(crate) fn check_slm_consistency(model: &Sequential, meta: &ModelMetadata) -> Result<()> {
    if meta.task != TaskType::TransformerSLM {
        return Ok(());
    }
    let layers = model.layers();
    let emb = layers
        .iter()
        .find_map(|l| l.as_any().downcast_ref::<Embedding>())
        .ok_or_else(|| bad("token-ID SLM has no embedding layer".into()))?;
    let head = layers
        .iter()
        .rev()
        .find_map(|l| l.as_any().downcast_ref::<Linear>())
        .ok_or_else(|| bad("token-ID SLM has no output projection".into()))?;
    let (seq, vocab) = (emb.max_seq_len(), emb.vocab_size());
    let mismatch = |what: &str, got: usize, want: usize| {
        Err(bad(format!(
            "model metadata disagrees with its layers: {what} {got}, layers say {want}"
        )))
    };
    if meta.input_dim != seq {
        return mismatch("input_dim", meta.input_dim, seq);
    }
    if meta.output_dim != vocab {
        return mismatch("output_dim", meta.output_dim, vocab);
    }
    if head.out_features() != vocab {
        return mismatch("LM head width", head.out_features(), vocab);
    }
    for l in layers {
        if let Some(b) = l.as_any().downcast_ref::<TransformerBlock>() {
            if b.context_len() != seq {
                return mismatch("block context", b.context_len(), seq);
            }
        }
    }
    let table = if meta.tokenizer_state.is_empty() {
        meta.class_names.len()
    } else {
        ByteBpeTokenizer::from_state(&meta.tokenizer_state)?.vocab_size()
    };
    if table != vocab {
        return mismatch("tokenizer vocabulary", table, vocab);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_slm_consistency;
    use crate::layer::{Embedding, Layer, Linear, TransformerBlock};
    use crate::loader::test_support::{make_bundle, make_embedding_bundle_big};
    use crate::meta::{ModelMetadata, TaskType};
    use crate::model::Sequential;
    use crate::tokenizer::ByteBpeTokenizer;

    #[test]
    fn slm_consistency_accepts_a_matching_model_and_names_each_mismatch() {
        let (m, _, meta) = make_embedding_bundle_big();
        check_slm_consistency(&m, &meta).unwrap();
        // Non-SLM tasks are not checked.
        let (tm, _, tmeta) = make_bundle();
        check_slm_consistency(&tm, &tmeta).unwrap();

        let err = |meta: ModelMetadata| check_slm_consistency(&m, &meta).unwrap_err().to_string();
        let mut bad_meta = meta.clone();
        bad_meta.input_dim = 0;
        assert!(err(bad_meta).contains("input_dim 0, layers say 4"));
        let mut bad_meta = meta.clone();
        bad_meta.input_dim = 100_000_000;
        assert!(err(bad_meta).contains("input_dim"));
        let mut bad_meta = meta.clone();
        bad_meta.output_dim = 11;
        assert!(err(bad_meta).contains("output_dim 11, layers say 10"));
        let mut bad_meta = meta.clone();
        bad_meta.class_names.pop();
        assert!(err(bad_meta).contains("tokenizer vocabulary 9, layers say 10"));
        let mut bad_meta = meta.clone();
        bad_meta.tokenizer_state = "not,a,merge".into();
        assert!(check_slm_consistency(&m, &bad_meta).is_err());
        let mut bpe_meta = meta.clone();
        bpe_meta.tokenizer_state = String::new();
        bpe_meta.class_names.clear();
        assert!(err(bpe_meta).contains("tokenizer vocabulary 0, layers say 10"));
    }

    #[test]
    fn slm_consistency_checks_bpe_vocab_head_width_and_block_context() {
        let tok = ByteBpeTokenizer::train(&"abc abd abe ".repeat(30), 270).unwrap();
        let v = tok.vocab_size();
        let emb =
            |seq: usize| Embedding::new(v, seq, 4, vec![0.1; v * 4], vec![0.1; seq * 4]).unwrap();
        let head = |w: usize| Linear::new(4, w, vec![0.1; 4 * w], vec![0.0; w]).unwrap();
        let meta = ModelMetadata {
            dataset_name: "b".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: vec![],
            target_name: "".into(),
            target_range: [0.0, 0.0],
            input_dim: 4,
            output_dim: v,
            tokenizer_state: tok.encode_state(),
        };
        let seq = |layers: Vec<Box<dyn Layer>>| {
            layers.into_iter().fold(Sequential::new(), |m, l| m.with(l))
        };
        check_slm_consistency(&seq(vec![Box::new(emb(4)), Box::new(head(v))]), &meta).unwrap();
        // A tokenizer with a different vocabulary than the layers.
        let mut other = meta.clone();
        other.tokenizer_state = ByteBpeTokenizer::train(&"xyz xyw ".repeat(30), 262)
            .unwrap()
            .encode_state();
        let e = check_slm_consistency(&seq(vec![Box::new(emb(4)), Box::new(head(v))]), &other)
            .unwrap_err();
        assert!(e.to_string().contains("tokenizer vocabulary"), "{e}");
        // LM head narrower than the vocabulary.
        let e = check_slm_consistency(&seq(vec![Box::new(emb(4)), Box::new(head(v - 1))]), &meta)
            .unwrap_err();
        assert!(e.to_string().contains("LM head width"), "{e}");
        // No embedding / no output projection.
        let e = check_slm_consistency(&seq(vec![Box::new(head(v))]), &meta).unwrap_err();
        assert!(e.to_string().contains("no embedding"), "{e}");
        let e = check_slm_consistency(&seq(vec![Box::new(emb(4))]), &meta).unwrap_err();
        assert!(e.to_string().contains("no output projection"), "{e}");
        // A block whose context differs from the embedding's.
        let c = 4usize;
        let block = TransformerBlock::new(
            8,
            2,
            c,
            vec![1.0; c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![1.0; c],
            vec![0.0; c],
            vec![0.1; c * 8],
            vec![0.0; 8],
            vec![0.1; 8 * c],
            vec![0.0; c],
        )
        .unwrap();
        let e = check_slm_consistency(
            &seq(vec![Box::new(emb(4)), Box::new(block), Box::new(head(v))]),
            &meta,
        )
        .unwrap_err();
        assert!(
            e.to_string().contains("block context 8, layers say 4"),
            "{e}"
        );
    }
}
