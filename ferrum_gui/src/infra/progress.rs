//! Throttled progress for the long commands that stream (downloads, Clean): at most one
//! event per [`INTERVAL_MS`], plus a final one that always goes out. The clock is a
//! closure so tests drive it by hand (no sleeping, no `Instant::now`).

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
