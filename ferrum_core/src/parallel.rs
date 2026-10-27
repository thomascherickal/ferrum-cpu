//! Dynamic CPU parallelism for the hot numeric kernels — built only on `std`.
//!
//! Ferrum adds no threading crates and is `#![forbid(unsafe_code)]`, so
//! there is no `rayon`/`num_cpus` and no `unsafe` lifetime tricks. Instead this
//! module runs a **persistent worker pool**: a fixed set of `'static` threads is
//! spawned once (lazily, on first use) and reused for every matmul, rather than
//! creating fresh OS threads per call. This matters most for autoregressive
//! generation, which issues thousands of small matmuls whose per-call
//! thread-creation cost would otherwise dominate (and even regress at high
//! thread counts).
//!
//! Because safe Rust cannot hand a borrowed closure to threads that outlive the
//! call, kernels share their read-only inputs through [`std::sync::Arc`] (one
//! cheap clone per matmul, not per worker) and each worker computes and returns
//! an **owned** output block; the caller stitches the blocks back together.
//!
//! Large matmuls skip the pool: [`run_scoped`] starts scoped threads that
//! *borrow* the operands and write straight into the output, which is cheaper
//! than copying everything once the multiply dominates the spawn ([`SCOPED_MIN_COST`]).
//!
//! The design is opt-out-safe:
//!
//! * Workloads below a scalar-work threshold run serially on the calling thread
//!   ([`should_parallelize`] returns `false`), so small matmuls pay nothing.
//! * The `wasm32` target, which has no threads, always runs serially and never
//!   spawns the pool.
//! * Splitting the output by rows never changes the per-element arithmetic, so
//!   results are **bit-for-bit identical** regardless of the thread count and
//!   training/inference stay deterministic.
//!
//! GPUs are never used: all parallelism is plain CPU threads.

#[cfg(not(target_arch = "wasm32"))]
use std::panic::{catch_unwind, AssertUnwindSafe};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{channel, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::sync::OnceLock;

static NUM_THREADS: OnceLock<usize> = OnceLock::new();

/// The worker-thread count Ferrum uses for parallel kernels, detected once and
/// cached for the lifetime of the process.
///
/// Resolution order:
/// 1. `FERRUM_NUM_THREADS` environment variable, if set to a positive integer.
/// 2. [`std::thread::available_parallelism`] (the CPU's reported parallelism).
/// 3. `1` if neither is available.
///
/// Always `1` on `wasm32`, which has no threads.
pub fn num_threads() -> usize {
    *NUM_THREADS.get_or_init(detect)
}

#[cfg(target_arch = "wasm32")]
fn detect() -> usize {
    1
}

#[cfg(not(target_arch = "wasm32"))]
fn detect() -> usize {
    if let Ok(v) = std::env::var("FERRUM_NUM_THREADS") {
        match parse_thread_env(&v) {
            Some(n) => return n,
            None => eprintln!(
                "warning: FERRUM_NUM_THREADS={v:?} is not a positive integer; using the detected core count"
            ),
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// A `FERRUM_NUM_THREADS` value: `Some(n)` for a positive integer, `None` otherwise.
#[cfg(not(target_arch = "wasm32"))]
fn parse_thread_env(v: &str) -> Option<usize> {
    v.parse::<usize>().ok().filter(|&n| n >= 1)
}

/// Total scalar multiply–add count below which threading is not worth its
/// dispatch cost; smaller outputs are computed serially.
const PARALLEL_THRESHOLD: usize = 1 << 16;

/// Whether a row-major output with `rows` rows and an estimated total scalar
/// `cost` (e.g. `m * k * n` for a matmul) is worth splitting across the worker
/// pool. Callers should run the serial path directly when this is `false` to
/// avoid the input `Arc` clone the parallel path needs.
pub fn should_parallelize(rows: usize, cost: usize) -> bool {
    rows >= 2 && cost >= PARALLEL_THRESHOLD && num_threads() > 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent worker pool
// ─────────────────────────────────────────────────────────────────────────────

/// Panic message of a caller whose worker job panicked (see the pool loop).
#[cfg(not(target_arch = "wasm32"))]
const WORKER_PANICKED: &str = "ferrum: a parallel worker job panicked";

#[cfg(not(target_arch = "wasm32"))]
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed set of long-lived worker threads, each owning one job channel.
#[cfg(not(target_arch = "wasm32"))]
struct Pool {
    senders: Vec<Sender<Job>>,
}

#[cfg(not(target_arch = "wasm32"))]
static POOL: OnceLock<Pool> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        build_pool(num_threads().max(1), |id, rx| {
            std::thread::Builder::new()
                .name(format!("ferrum-worker-{id}"))
                .spawn(move || {
                    // Park on the channel; run jobs until the sender is dropped
                    // (only at process shutdown, since `POOL` is 'static). A
                    // panicking job must not kill its worker: the job's result
                    // sender is dropped by the unwind, so the caller's `recv`
                    // fails (and the caller panics), but the pool stays whole
                    // for every later call.
                    while let Ok(job) = rx.recv() {
                        let _ = catch_unwind(AssertUnwindSafe(job));
                    }
                })
                .map(drop)
        })
    })
}

/// Start up to `n` workers with `spawn`. A failed spawn (e.g. `RLIMIT_NPROC`)
/// stops the build and keeps the workers started so far, so the pool degrades
/// to fewer workers - or none, in which case callers run serially - instead of
/// panicking. Results never depend on the worker count.
#[cfg(not(target_arch = "wasm32"))]
fn build_pool(
    n: usize,
    mut spawn: impl FnMut(usize, std::sync::mpsc::Receiver<Job>) -> std::io::Result<()>,
) -> Pool {
    let mut senders = Vec::with_capacity(n);
    for id in 0..n {
        let (tx, rx) = channel::<Job>();
        if spawn(id, rx).is_err() {
            break;
        }
        senders.push(tx);
    }
    Pool { senders }
}

/// Fill a row-major `[m, n]` output by running `kernel(r0, r1, block)` over
/// contiguous row blocks on the persistent worker pool.
///
/// `kernel` must fill rows `r0..r1` into `block`, a fresh buffer of exactly
/// `(r1 - r0) * n` elements indexed locally (row `i` lives at `(i - r0) * n`).
/// Every row is produced by exactly one call, so the result does not depend on
/// the number of threads. `kernel` is shared across workers via `Arc`, so it
/// must own its inputs (typically `Arc<[f32]>` clones) — that ownership is what
/// lets the work run on threads outliving this call without any `unsafe`.
///
/// Intended to be called only when [`should_parallelize`] is `true`; it is
/// always correct, and on `wasm32` simply runs the kernel serially.
pub fn run<K>(m: usize, n: usize, kernel: K) -> Vec<f32>
where
    K: Fn(usize, usize, &mut [f32]) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let mut out = vec![0.0f32; m * n];
        kernel(0, m, &mut out);
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = num_threads().max(1).min(m.max(1));
        // No workers (single thread, or every spawn failed): run serially.
        let pool = (threads > 1).then(pool).filter(|p| !p.senders.is_empty());
        let Some(pool) = pool else {
            let mut out = vec![0.0f32; m * n];
            kernel(0, m, &mut out);
            return out;
        };

        let rows_per = m.div_ceil(threads);
        let kernel = Arc::new(kernel);
        let (tx, rx) = channel::<(usize, Vec<f32>)>();

        let mut blocks = 0usize;
        let mut r0 = 0usize;
        while r0 < m {
            let r1 = (r0 + rows_per).min(m);
            let kernel = Arc::clone(&kernel);
            let tx = tx.clone();
            let job: Job = Box::new(move || {
                let mut block = vec![0.0f32; (r1 - r0) * n];
                kernel(r0, r1, &mut block);
                let _ = tx.send((r0, block));
            });
            // `blocks` never exceeds `threads`; if fewer workers started, they queue.
            let _ = pool.senders[blocks % pool.senders.len()].send(job);
            blocks += 1;
            r0 = r1;
        }
        drop(tx);

        let mut out = vec![0.0f32; m * n];
        for _ in 0..blocks {
            let (r0, block) = rx.recv().expect(WORKER_PANICKED);
            let start = r0 * n;
            out[start..start + block.len()].copy_from_slice(&block);
        }
        out
    }
}

/// Scalar multiply-add count from which a matmul is run by [`run_scoped`] (fresh
/// scoped threads that borrow their inputs) instead of by the persistent pool
/// ([`run`]).
///
/// The pool needs owned inputs, so every call copies `A` and `B` into `Arc`s and
/// every worker returns an owned block that the caller copies into the output. A
/// scoped call borrows `A` and `B` and lets each thread write its rows straight
/// into the output, but it pays for spawning the threads. Per-call interleaved
/// A/B runs (Manual §2.2.1) put the break-even at 17 to 25 million multiply-adds:
/// below it the pool wins (scoped +15% to +27% at 2.4 M), above it scoped calls
/// are faster (median -8% to -27% from 33 M). The threshold keeps every
/// measured break-even and loss on the pool side.
pub const SCOPED_MIN_COST: usize = 20_000_000;

/// Whether a matmul of `cost` multiply-adds should use [`run_scoped`].
pub fn prefers_scoped(cost: usize) -> bool {
    cost >= SCOPED_MIN_COST
}

/// Fill a row-major `[m, n]` output like [`run`], but on scoped threads that
/// **borrow** their inputs: no copy of the operands, and every thread writes its
/// rows straight into the output instead of returning a block to stitch.
/// `kernel(r0, r1, block)` is the contract of [`run`] (`block` is a zeroed slice
/// of `(r1 - r0) * n` elements indexed locally).
///
/// The row split is `ceil(m / threads)` rows per block, as in [`run`], and every
/// block is computed once, so the result is **bit-for-bit identical** at any
/// thread count. The caller works on the blocks too and `threads - 1` helper
/// threads are started; one that cannot be spawned just leaves its share to the
/// others, so the call never fails for lack of threads.
pub fn run_scoped<K>(m: usize, n: usize, kernel: K) -> Vec<f32>
where
    K: Fn(usize, usize, &mut [f32]) + Sync,
{
    run_scoped_with(num_threads(), m, n, kernel)
}

/// [`run_scoped`] with an explicit thread count (clamped to `1..=m`); the tests
/// use it to check the result at every thread count.
pub(crate) fn run_scoped_with<K>(threads: usize, m: usize, n: usize, kernel: K) -> Vec<f32>
where
    K: Fn(usize, usize, &mut [f32]) + Sync,
{
    let mut out = vec![0.0f32; m * n];
    let threads = threads.clamp(1, m.max(1));
    if threads == 1 || n == 0 || cfg!(target_arch = "wasm32") {
        kernel(0, m, &mut out);
        return out;
    }
    let rows_per = m.div_ceil(threads);
    let work: Vec<(usize, usize, &mut [f32])> = out
        .chunks_mut(rows_per * n)
        .enumerate()
        .map(|(i, block)| (i * rows_per, ((i + 1) * rows_per).min(m), block))
        .collect();
    let helpers = work.len() - 1;
    let work = std::sync::Mutex::new(work);
    let drain = || loop {
        // The lock is released before the kernel runs, so a panicking kernel
        // cannot poison it.
        let next = work.lock().unwrap_or_else(|e| e.into_inner()).pop();
        match next {
            Some((r0, r1, block)) => kernel(r0, r1, block),
            None => break,
        }
    };
    std::thread::scope(|scope| {
        for _ in 0..helpers {
            if std::thread::Builder::new()
                .spawn_scoped(scope, drain)
                .is_err()
            {
                break;
            }
        }
        drain();
    });
    out
}

/// Fill a length-`total` 1-D output by running `kernel(j0, j1, block)` over
/// contiguous column ranges on the persistent worker pool, where `block` is a
/// fresh buffer of `j1 - j0` elements indexed locally (`j` at `j - j0`).
///
/// This is the column-split counterpart of [`run`]: where `run` parallelizes a
/// `[m, n]` output by **rows** (needs `m ≥ 2`, useless for single-token decode),
/// this parallelizes a single output vector by **columns**. It is what lets the
/// autoregressive GEMV — every decode matmul has `m = 1` — use more than one
/// core. Because each output index is produced by exactly one worker and the
/// reduction over the contraction dimension happens entirely inside that worker,
/// the result is **bit-for-bit identical** regardless of the thread count.
///
/// `min_chunk` keeps workers from being handed trivially small slices; the whole
/// thing runs serially below it or when threading is unavailable.
pub fn run_1d<K>(total: usize, min_chunk: usize, kernel: K) -> Vec<f32>
where
    K: Fn(usize, usize, &mut [f32]) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = min_chunk;
        let mut out = vec![0.0f32; total];
        kernel(0, total, &mut out);
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = num_threads()
            .max(1)
            .min(total.div_ceil(min_chunk.max(1)).max(1));
        // No workers (single thread, or every spawn failed): run serially.
        let pool = (threads > 1).then(pool).filter(|p| !p.senders.is_empty());
        let Some(pool) = pool else {
            let mut out = vec![0.0f32; total];
            kernel(0, total, &mut out);
            return out;
        };

        let cols_per = total.div_ceil(threads);
        let kernel = Arc::new(kernel);
        let (tx, rx) = channel::<(usize, Vec<f32>)>();

        let mut blocks = 0usize;
        let mut j0 = 0usize;
        while j0 < total {
            let j1 = (j0 + cols_per).min(total);
            let kernel = Arc::clone(&kernel);
            let tx = tx.clone();
            let job: Job = Box::new(move || {
                let mut block = vec![0.0f32; j1 - j0];
                kernel(j0, j1, &mut block);
                let _ = tx.send((j0, block));
            });
            let _ = pool.senders[blocks % pool.senders.len()].send(job);
            blocks += 1;
            j0 = j1;
        }
        drop(tx);

        let mut out = vec![0.0f32; total];
        for _ in 0..blocks {
            let (j0, block) = rx.recv().expect(WORKER_PANICKED);
            out[j0..j0 + block.len()].copy_from_slice(&block);
        }
        out
    }
}

/// Whether a 1-D output of `total` elements with `per_element` scalar work each
/// is worth splitting across the pool (the GEMV/decode gate).
pub fn should_parallelize_1d(total: usize, per_element: usize) -> bool {
    total >= 2 && total.saturating_mul(per_element) >= PARALLEL_THRESHOLD && num_threads() > 1
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn thread_env_accepts_only_positive_integers() {
        use super::parse_thread_env as p;
        assert_eq!((p("4"), p("8")), (Some(4), Some(8)));
        for bad in ["abc", "0", "-2", "", "1.5", "4x", " 8 "] {
            assert_eq!(p(bad), None, "{bad:?}");
        }
    }

    use super::*;

    #[test]
    fn num_threads_is_at_least_one() {
        assert!(num_threads() >= 1);
    }

    #[test]
    fn run_covers_every_row_exactly_once() {
        // A large output (above the threshold) split across the pool must fill
        // every element exactly once, regardless of how many workers ran it.
        let (m, n) = (257usize, 129usize);
        let out = run(m, n, move |r0, r1, block| {
            for i in r0..r1 {
                let o = (i - r0) * n;
                for j in 0..n {
                    block[o + j] = (i * n + j) as f32;
                }
            }
        });
        assert_eq!(out.len(), m * n);
        for (idx, &v) in out.iter().enumerate() {
            assert_eq!(v, idx as f32, "element {idx} not written exactly once");
        }
    }

    #[test]
    fn run_scoped_covers_every_row_exactly_once_at_any_thread_count() {
        for (m, n) in [
            (0usize, 4usize),
            (1, 3),
            (2, 5),
            (7, 3),
            (64, 1),
            (257, 129),
        ] {
            for threads in 1..=8 {
                let out = run_scoped_with(threads, m, n, |r0, r1, block| {
                    assert_eq!(block.len(), (r1 - r0) * n);
                    assert!(block.iter().all(|&v| v == 0.0), "block starts zeroed");
                    for i in r0..r1 {
                        for j in 0..n {
                            block[(i - r0) * n + j] = (i * n + j) as f32;
                        }
                    }
                });
                assert_eq!(out.len(), m * n);
                for (idx, &v) in out.iter().enumerate() {
                    assert_eq!(v, idx as f32, "{m}x{n} threads={threads} element {idx}");
                }
            }
        }
        // A zero-width output must not panic in the row split.
        assert!(run_scoped_with(4, 5, 0, |_, _, _| {}).is_empty());
    }

    #[test]
    fn prefers_scoped_switches_at_the_threshold() {
        assert!(!prefers_scoped(SCOPED_MIN_COST - 1));
        assert!(prefers_scoped(SCOPED_MIN_COST));
    }

    #[test]
    fn should_parallelize_gates_on_size() {
        assert!(
            !should_parallelize(1, usize::MAX),
            "single row stays serial"
        );
        assert!(!should_parallelize(1000, 0), "trivial work stays serial");
        if num_threads() > 1 {
            assert!(should_parallelize(1000, PARALLEL_THRESHOLD));
        }
    }

    #[test]
    fn run_1d_covers_every_column_exactly_once() {
        let total = 1000usize;
        let out = run_1d(total, 64, move |j0, j1, block| {
            for (idx, j) in (j0..j1).enumerate() {
                block[idx] = (j * 2) as f32;
            }
        });
        assert_eq!(out.len(), total);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, (j * 2) as f32, "column {j} not written exactly once");
        }
    }

    #[test]
    fn run_1d_serial_when_min_chunk_exceeds_total() {
        // min_chunk ≥ total collapses to a single serial invocation.
        let out = run_1d(50, 1_000_000, move |j0, j1, block| {
            for (idx, j) in (j0..j1).enumerate() {
                block[idx] = j as f32;
            }
        });
        assert_eq!(out.len(), 50);
        assert!(out.iter().enumerate().all(|(j, &v)| v == j as f32));
    }

    #[test]
    fn should_parallelize_1d_gates() {
        assert!(
            !should_parallelize_1d(1, usize::MAX),
            "single element stays serial"
        );
        assert!(!should_parallelize_1d(1000, 0), "no work stays serial");
        if num_threads() > 1 {
            assert!(should_parallelize_1d(PARALLEL_THRESHOLD, 1));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn build_pool_degrades_when_spawn_fails() {
        let ok = |_: usize, _: std::sync::mpsc::Receiver<Job>| Ok(());
        assert_eq!(build_pool(4, ok).senders.len(), 4);
        let flaky = |id: usize, _: std::sync::mpsc::Receiver<Job>| {
            if id < 2 {
                Ok(())
            } else {
                Err(std::io::Error::other("EAGAIN"))
            }
        };
        assert_eq!(build_pool(4, flaky).senders.len(), 2);
        let none = |_: usize, _: std::sync::mpsc::Receiver<Job>| Err(std::io::Error::other("x"));
        assert!(build_pool(4, none).senders.is_empty());
    }

    /// Regression: a panicking kernel used to kill its pool worker for good, so
    /// every later parallel call on that worker failed with "worker thread
    /// disconnected" (a long-lived server then errored on every request).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn worker_panic_fails_that_call_but_not_the_pool() {
        let boom = |f: &dyn Fn() -> Vec<f32>| {
            let r = catch_unwind(AssertUnwindSafe(f));
            let msg = r.unwrap_err();
            msg.downcast_ref::<String>().cloned()
        };
        // Enough rounds that every worker has run (and survived) a panicking job.
        for _ in 0..(2 * num_threads() + 2) {
            let m = boom(&|| run(512, 4, |_, _, _| panic!("kernel boom")));
            let m1 = boom(&|| run_1d(4096, 1, |_, _, _| panic!("kernel boom")));
            if num_threads() > 1 {
                assert!(m.unwrap().contains(WORKER_PANICKED));
                assert!(m1.unwrap().contains(WORKER_PANICKED));
            }
        }
        let out = run(257, 3, |r0, r1, b| {
            for i in r0..r1 {
                b[(i - r0) * 3..(i - r0) * 3 + 3].fill(i as f32);
            }
        });
        assert!((0..257).all(|i| out[i * 3..i * 3 + 3].iter().all(|&v| v == i as f32)));
        let out = run_1d(1000, 1, |j0, j1, b| {
            for (k, j) in (j0..j1).enumerate() {
                b[k] = j as f32;
            }
        });
        assert!(out.iter().enumerate().all(|(j, &v)| v == j as f32));
    }
}
