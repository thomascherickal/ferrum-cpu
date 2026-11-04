//! The system monitor: CPU and memory load, polled by the frontend.

use crate::infra::error::CmdError;
use crate::AppState;
use serde::Serialize;
use tauri::State;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SysStats {
    pub available: bool,
    pub cpu_total: f32,
    pub per_core: Vec<f32>,
    pub cores: usize,
    pub mem_used: u64,
    pub mem_total: u64,
    pub mem_percent: f32,
    pub ferrum_threads: usize,
}

/// Sample current CPU and memory load. Polled by the frontend on an interval.
#[tauri::command]
pub fn system_stats(state: State<'_, AppState>) -> Result<SysStats, CmdError> {
    // A poisoned lock only means an earlier poll panicked; sysinfo state is still valid.
    let mut sys = state.sys.lock().unwrap_or_else(|e| e.into_inner());
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let mem_total = sys.total_memory();
    let mem_used = sys.used_memory();
    let mem_percent = percent(mem_used, mem_total);
    Ok(SysStats {
        available: true,
        cpu_total: sys.global_cpu_usage(),
        cores: per_core.len(),
        per_core,
        mem_used,
        mem_total,
        mem_percent,
        ferrum_threads: ferrum_core::num_threads(),
    })
}

/// `used` as a percentage of `total`; 0 when the total is unknown.
fn percent(used: u64, total: u64) -> f32 {
    if total > 0 {
        used as f32 / total as f32 * 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::percent;

    #[test]
    fn memory_percent_is_zero_when_the_total_is_unknown() {
        assert_eq!(percent(50, 200), 25.0);
        assert_eq!(percent(50, 0), 0.0);
    }
}
