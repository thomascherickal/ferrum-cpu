//! Throttled progress for the long commands that stream (downloads, Clean): at most one
//! event per [`INTERVAL_MS`], plus a final one that always goes out. The clock is a
//! closure so tests drive it by hand (no sleeping, no `Instant::now`). [`Progress`] and
//! [`Tap`] turn a download's byte stream into `download-progress` events.

use std::io::Read;
use tauri::{AppHandle, Emitter, Runtime};

/// Smallest gap between two throttled events, in milliseconds.
pub const INTERVAL_MS: u64 = 100;

/// Lets an event through when none went out yet or `INTERVAL_MS` has passed since the last.
pub struct Throttle<'a> {
    clock: Box<dyn FnMut() -> u64 + 'a>,
    last: Option<u64>,
}

impl<'a> Throttle<'a> {
    /// `clock` returns the elapsed milliseconds (any monotonic source).
    pub fn new(clock: impl FnMut() -> u64 + 'a) -> Self {
        Throttle {
            clock: Box::new(clock),
            last: None,
        }
    }

    /// A throttle on the real clock, counting from now.
    pub fn wall() -> Throttle<'static> {
        let t0 = std::time::Instant::now();
        Throttle::new(move || t0.elapsed().as_millis() as u64)
    }

    /// True when an event may go out now (and records that it did).
    pub fn ready(&mut self) -> bool {
        let now = (self.clock)();
        if self
            .last
            .is_some_and(|l| now.saturating_sub(l) < INTERVAL_MS)
        {
            return false;
        }
        self.last = Some(now);
        true
    }
}

/// Download progress: `update` is throttled to one event per 100 ms, `finish` always goes
/// out. The sink receives `(bytes so far, total if the server said)`.
pub(crate) struct Progress<'a> {
    throttle: Throttle<'a>,
    sink: Box<dyn FnMut(u64, Option<u64>) + 'a>,
}

impl<'a> Progress<'a> {
    pub(crate) fn new(throttle: Throttle<'a>, sink: impl FnMut(u64, Option<u64>) + 'a) -> Self {
        Progress {
            throttle,
            sink: Box::new(sink),
        }
    }

    /// Reports nothing.
    #[cfg(test)]
    pub(crate) fn silent() -> Progress<'static> {
        Progress::new(Throttle::wall(), |_, _| {})
    }

    /// Emits `download-progress {id, bytes, total}` for `id` on the real clock.
    pub(crate) fn events<R: Runtime>(app: AppHandle<R>, id: String) -> Progress<'static> {
        Progress::new(Throttle::wall(), move |bytes, total| {
            let _ = app.emit(
                "download-progress",
                serde_json::json!({ "id": id, "bytes": bytes, "total": total }),
            );
        })
    }

    pub(crate) fn update(&mut self, bytes: u64, total: Option<u64>) {
        if self.throttle.ready() {
            (self.sink)(bytes, total);
        }
    }

    pub(crate) fn finish(&mut self, bytes: u64) {
        (self.sink)(bytes, Some(bytes));
    }
}

/// A reader that reports the running byte count after every read.
pub(crate) struct Tap<'p, 'a, R> {
    inner: R,
    bytes: u64,
    total: Option<u64>,
    progress: &'p mut Progress<'a>,
}

impl<'p, 'a, R> Tap<'p, 'a, R> {
    /// Wraps `inner`; `total` is the size the server announced, if any.
    pub(crate) fn new(inner: R, total: Option<u64>, progress: &'p mut Progress<'a>) -> Self {
        Tap {
            inner,
            bytes: 0,
            total,
            progress,
        }
    }
}

impl<R: Read> Read for Tap<'_, '_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        self.progress.update(self.bytes, self.total);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_burst_lets_one_event_through_per_window() {
        let mut t = 0u64;
        let mut th = Throttle::new(|| {
            t += 30; // each check is 30 ms after the last
            t
        });
        // Checks at 30, 60, 90, 120, 150, 180, 210, 240 ms.
        let got: Vec<bool> = (0..8).map(|_| th.ready()).collect();
        // 30 passes (first); 60, 90, 120 are under 100 ms later; 150 passes; 180..240 do not.
        assert_eq!(
            got,
            [true, false, false, false, true, false, false, false],
            "{got:?}"
        );
    }

    #[test]
    fn exactly_one_interval_apart_passes_and_a_clock_going_back_does_not_panic() {
        let times = [0u64, 99, 100, 50];
        let mut i = 0;
        let mut th = Throttle::new(|| {
            i += 1;
            times[i - 1]
        });
        assert_eq!(
            [th.ready(), th.ready(), th.ready(), th.ready()],
            [true, false, true, false]
        );
    }

    #[test]
    fn the_wall_clock_throttle_passes_its_first_event() {
        let mut th = Throttle::wall();
        assert!(th.ready());
    }
}
