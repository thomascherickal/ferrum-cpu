//! Per-step training metrics for drivers (CSV logs, progress lines). Everything here is off by
//! default: with no tracking requested the step does exactly the arithmetic it did before.
use crate::train_transformer::TransformerNet;

/// What the last optimizer step measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepMetrics {
    /// Mean training loss of the step's minibatch.
    pub loss: f32,
    /// Global L2 gradient norm before clipping. `None` unless clipping is on or
    /// [`TransformerNet::set_track_grad_norm`] was called (the norm costs one pass over the gradients).
    pub grad_norm: Option<f32>,
}

#[derive(Clone, Default)]
pub(super) struct MetricsState {
    track_grad_norm: bool,
    last: Option<StepMetrics>,
}

impl TransformerNet {
    /// Measure the gradient norm every step even without clipping. Off by default.
    pub fn set_track_grad_norm(&mut self, on: bool) {
        self.metrics.track_grad_norm = on;
    }

    /// The metrics of the most recent step of the built-in epoch helpers
    /// ([`train_transformer_steps`](crate::train_transformer_steps)); read it from the step hook.
    pub fn last_step_metrics(&self) -> Option<StepMetrics> {
        self.metrics.last
    }

    /// Clip the gradients as configured and record this step's metrics.
    pub(super) fn finish_grads(&mut self, loss: f32) {
        let grad_norm = match self.grad_clip {
            Some(max_norm) => Some(self.clip_grad_norm(max_norm)),
            None if self.metrics.track_grad_norm => {
                // A zero budget never rescales: this only measures.
                Some(crate::optim::clip_grad_norm(
                    &mut self.grad_tensors_mut(),
                    0.0,
                ))
            }
            None => None,
        };
        self.metrics.last = Some(StepMetrics { loss, grad_norm });
    }
}

#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::train_transformer::{train_transformer_steps, TransformerNet};

    fn run(track: bool, threads: usize) -> (Vec<f32>, Vec<Option<f32>>, Vec<f32>) {
        let mut net = TransformerNet::new(5, 4, 8, 2, 16, 1, &mut Rng::new(3)).unwrap();
        net.set_window_stride(4);
        net.set_track_grad_norm(track);
        let tokens: Vec<usize> = (0..120).map(|i| (i * 3) % 5).collect();
        let (mut losses, mut norms) = (Vec::new(), Vec::new());
        train_transformer_steps(
            &mut net,
            &tokens,
            4,
            &Adam::new(0.01),
            &mut Rng::new(9),
            threads,
            usize::MAX,
            &mut |n, _| {
                let m = n.last_step_metrics().unwrap();
                losses.push(m.loss);
                norms.push(m.grad_norm);
                Ok(true)
            },
        )
        .unwrap();
        (losses, norms, net.snapshot_weights().concat())
    }

    #[test]
    fn grad_norm_is_measured_only_on_request_and_never_changes_the_weights() {
        for threads in [1, 2] {
            let (l0, n0, w0) = run(false, threads);
            let (l1, n1, w1) = run(true, threads);
            assert!(n0.iter().all(Option::is_none), "off by default");
            assert!(n1
                .iter()
                .all(|n| n.is_some_and(|x| x.is_finite() && x > 0.0)));
            assert_eq!(l0, l1);
            assert_eq!(
                w0, w1,
                "tracking must not change the trained weights (threads {threads})"
            );
            assert!(l1.len() > 3);
        }
    }

    #[test]
    fn clipping_reports_the_pre_clip_norm() {
        let mut net = TransformerNet::new(5, 4, 8, 2, 16, 1, &mut Rng::new(3)).unwrap();
        net.set_grad_clip(Some(1e-4));
        let tokens: Vec<usize> = (0..60).map(|i| i % 5).collect();
        let mut seen = Vec::new();
        train_transformer_steps(
            &mut net,
            &tokens,
            4,
            &Adam::new(0.01),
            &mut Rng::new(1),
            1,
            2,
            &mut |n, _| {
                seen.push(n.last_step_metrics().unwrap().grad_norm.unwrap());
                Ok(true)
            },
        )
        .unwrap();
        assert!(seen.iter().all(|&n| n > 1e-4), "{seen:?}");
    }
}
