//! Shared fixtures for the unit tests of this module tree.
use crate::loss::softmax_cross_entropy;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::train_transformer::epoch::train_transformer_epoch;
use crate::train_transformer::TransformerNet;

pub(super) fn tiny_net(rng: &mut Rng) -> TransformerNet {
    // vocab=5, T=3, C=4, heads=2, hidden=6, 1 block
    TransformerNet::new(5, 3, 4, 2, 6, 1, rng).unwrap()
}

pub(super) fn loss_of(net: &TransformerNet, tokens: &[usize], targets: &[usize]) -> f32 {
    let (logits, _) = net.forward(tokens).unwrap();
    softmax_cross_entropy(&logits, targets).unwrap().0
}

/// Finite-difference check of one parameter tensor against analytic grads.
pub(super) fn check_param<G, S>(
    net: &mut TransformerNet,
    tokens: &[usize],
    targets: &[usize],
    get_grad: G,
    set_data: S,
    indices: &[usize],
    label: &str,
) where
    G: Fn(&TransformerNet, usize) -> f32,
    S: Fn(&mut TransformerNet, usize, f32),
{
    let (logits, cache) = net.forward(tokens).unwrap();
    let (_, dl) = softmax_cross_entropy(&logits, targets).unwrap();
    net.zero_grad();
    net.backward(&dl, &cache).unwrap();

    let eps = 1e-2f32;
    for &i in indices {
        let analytic = get_grad(net, i);
        // Central difference: nudge +eps, then −2eps, then restore.
        set_data(net, i, eps);
        let lp = loss_of(net, tokens, targets);
        set_data(net, i, -2.0 * eps);
        let lm = loss_of(net, tokens, targets);
        set_data(net, i, eps);
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (numeric - analytic).abs() < 6e-3 + 0.03 * analytic.abs(),
            "{label}[{i}]: analytic={analytic:.6} numeric={numeric:.6}"
        );
    }
}

pub(super) fn trained_net(tie: bool, qat: bool) -> (TransformerNet, Rng) {
    let mut r = Rng::new(3);
    let mut net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
    net.set_qat(qat);
    net.set_weight_tying(tie);
    let tokens: Vec<usize> = (0..100).map(|i| (i * 5) % 6).collect();
    let adam = Adam::new(0.01);
    let mut rng = Rng::new(11);
    train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
    (net, rng)
}

/// Rebuild a checkpoint's GGUF with `edit` applied to each tensor
/// `(name, dims, type)`; data is zero-filled to the *edited* extent, so
/// the result is always a structurally valid, non-overlapping GGUF.
pub(super) fn mangled(bytes: &[u8], edit: impl Fn(&mut (String, Vec<u64>, u32))) -> Vec<u8> {
    let g = crate::gguf::Gguf::parse(bytes.to_vec()).unwrap();
    let mut b = crate::gguf_write::GgufBuilder::new();
    for (k, v) in &g.metadata {
        b.meta(k, v.clone());
    }
    for t in &g.tensors {
        let mut e = (t.name.clone(), t.dims.clone(), t.ggml_type);
        edit(&mut e);
        let n: u64 = e.1.iter().product();
        let width = if e.2 == crate::gguf::GGML_F32 { 4 } else { 2 };
        b.tensor(&e.0, &e.1, e.2, vec![0u8; n as usize * width]);
    }
    b.into_bytes()
}
