//! `TransformerConfig::metrics_csv`: one CSV row per optimizer step, plus the ETA text for the
//! progress line.
use crate::error::Result;
use std::io::Write;

const HEADER: &str = "step,epoch,loss,lr,grad_norm,tokens_per_sec,elapsed\n";

pub(super) struct CsvLog {
    file: Box<dyn Write>,
    last_secs: f64,
}

pub(super) struct Row {
    pub step: u64,
    pub epoch: usize,
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: Option<f32>,
    /// Tokens trained since the previous row.
    pub tokens: f64,
    /// Seconds since the run started.
    pub elapsed: f64,
}

impl CsvLog {
    /// Create (or, when `append`, extend) the file; a header is written to an empty file.
    pub fn open(path: &str, append: bool) -> Result<Self> {
        let mut f = std::fs::OpenOptions::new();
        f.create(true).write(true);
        if append {
            f.append(true);
        } else {
            f.truncate(true);
        }
        let mut file = f.open(path)?;
        if file.metadata()?.len() == 0 {
            file.write_all(HEADER.as_bytes())?;
        }
        Ok(Self::to(Box::new(file)))
    }

    /// A log that writes its rows (no header) to `file`.
    pub fn to(file: Box<dyn Write>) -> Self {
        Self {
            file,
            last_secs: 0.0,
        }
    }

    pub fn write(&mut self, r: &Row) -> Result<()> {
        let dt = (r.elapsed - self.last_secs).max(1e-9);
        self.last_secs = r.elapsed;
        let norm = r.grad_norm.map_or(String::new(), |n| n.to_string());
        let line = format!(
            "{},{},{},{},{norm},{:.1},{:.3}\n",
            r.step,
            r.epoch,
            r.loss,
            r.lr,
            r.tokens / dt,
            r.elapsed
        );
        self.file.write_all(line.as_bytes())?;
        Ok(())
    }
}

/// `45s`, `3m05s`, `2h03m`.
pub(super) fn format_eta(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_formats_seconds_minutes_hours() {
        assert_eq!(format_eta(0.4), "0s");
        assert_eq!(format_eta(45.0), "45s");
        assert_eq!(format_eta(185.0), "3m05s");
        assert_eq!(format_eta(7380.0), "2h03m");
        assert_eq!(format_eta(-3.0), "0s");
    }

    #[test]
    fn csv_has_one_header_and_appends_on_resume() {
        let p = std::env::temp_dir().join(format!("ferrum_csv_{}.csv", std::process::id()));
        let path = p.to_str().unwrap();
        let row = |step| Row {
            step,
            epoch: 1,
            loss: 2.5,
            lr: 0.01,
            grad_norm: Some(0.5),
            tokens: 100.0,
            elapsed: step as f64,
        };
        let mut a = CsvLog::open(path, false).unwrap();
        a.write(&row(1)).unwrap();
        let mut b = CsvLog::open(path, true).unwrap();
        b.write(&row(2)).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], HEADER.trim());
        assert_eq!(lines[1], "1,1,2.5,0.01,0.5,100.0,1.000");
        assert_eq!(lines.len(), 3, "{text}");
        std::fs::remove_file(p).unwrap();
    }
}
