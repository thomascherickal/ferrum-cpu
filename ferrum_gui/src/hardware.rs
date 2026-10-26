//! Hardware facts for the GUI: CPU, RAM and (best effort) GPUs.
//!
//! The Tauri command [`hardware_info`] feeds the "Your hardware" line and the
//! time/RAM estimates in [`crate::estimate`]. CPU and RAM come from `sysinfo`
//! (already a dependency, and cross-platform), so there is nothing to
//! cfg-gate: where a value is unavailable it is `None`.
//!
//! GPUs are detected without any new dependency: `nvidia-smi` when it can be
//! spawned (2 s timeout), and `/sys/class/drm/card*/device/*` plus the AMD KFD
//! topology when readable (Linux only; elsewhere the directories do not exist
//! and the list stays empty). **Ferrum is CPU-only**: every GPU is reported
//! with `used_by_ferrum = false` and the UI says "detected, not used by Ferrum".
//!
//! Every parser is a pure function over text so it is tested with fixtures.

use serde::Serialize;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// One graphics adapter found on the machine.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    /// Dedicated memory in MiB, when the driver reports it.
    pub memory_mb: Option<u64>,
    /// CUDA cores / compute units. Only set when a file or tool really reports
    /// it (AMD KFD `cu_count`); never guessed from the product name.
    pub cores: Option<u32>,
    /// Always `false`: Ferrum runs on the CPU only.
    pub used_by_ferrum: bool,
}

/// What the machine offers, as shown in the GUI and used by the estimates.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    /// Size of Ferrum's matmul worker pool (`FERRUM_NUM_THREADS` or the core count).
    pub ferrum_threads: usize,
    pub ram_total_mb: Option<u64>,
    pub ram_available_mb: Option<u64>,
    pub gpus: Vec<GpuInfo>,
}

// ── Pure parsers ─────────────────────────────────────────────────────────────

/// Parse `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits`.
///
/// One GPU per line, `name, total_mib, used_mib`. The name is split off from the
/// right so a comma inside it survives. A line needs three fields and a numeric
/// (or `[N/A]`) total; anything else (driver error text, blank lines) is skipped.
pub fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|line| {
            let mut it = line.trim().rsplitn(3, ',');
            let _used = it.next()?;
            let total = it.next()?.trim();
            let name = it.next()?.trim();
            let memory_mb = match total {
                "[N/A]" | "N/A" => None,
                t => Some(t.parse::<u64>().ok()?),
            };
            (!name.is_empty()).then(|| GpuInfo {
                name: name.to_string(),
                memory_mb,
                cores: None,
                used_by_ferrum: false,
            })
        })
        .collect()
}

/// `0x8086\n` (sysfs `vendor` / `device`) to its number.
pub fn parse_hex_id(s: &str) -> Option<u32> {
    let t = s.trim();
    u32::from_str_radix(t.strip_prefix("0x").unwrap_or(t), 16).ok()
}

/// PCI vendor id to a display name.
pub fn vendor_name(id: u32) -> String {
    match id {
        0x10de => "NVIDIA".into(),
        0x1002 => "AMD".into(),
        0x8086 => "Intel".into(),
        0x1af4 => "Virtual (virtio)".into(),
        0x15ad => "Virtual (VMware)".into(),
        other => format!("Unknown vendor 0x{other:04x}"),
    }
}

/// `mem_info_vram_total` (bytes) to MiB.
pub fn parse_vram_mb(s: &str) -> Option<u64> {
    s.trim().parse::<u64>().ok().map(|b| b / (1024 * 1024))
}

/// Value of `key` in an AMD KFD node `properties` file (`key value` per line).
pub fn kfd_property(props: &str, key: &str) -> Option<u64> {
    props.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        (it.next()? == key).then(|| it.next()?.parse().ok())?
    })
}

// ── sysfs readers (root is injectable so tests use a temp tree) ─────────────

fn read(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok()
}

/// Compute units of the KFD node whose vendor/device ids match, if any.
fn kfd_compute_units(nodes: &Path, vendor: u32, device: u32) -> Option<u32> {
    std::fs::read_dir(nodes).ok()?.flatten().find_map(|n| {
        let props = read(&n.path().join("properties"))?;
        let same = kfd_property(&props, "vendor_id")? == u64::from(vendor)
            && kfd_property(&props, "device_id")? == u64::from(device);
        let cu = kfd_property(&props, "cu_count").filter(|&c| c > 0)?;
        same.then_some(u32::try_from(cu).ok()?)
    })
}

/// Adapters under `<drm>/card<N>` (connector entries like `card0-DP-1` are not
/// adapters). Returns `(pci_vendor, info)` in card order.
pub fn sysfs_gpus(drm: &Path, kfd_nodes: &Path) -> Vec<(u32, GpuInfo)> {
    let mut cards: Vec<_> = std::fs::read_dir(drm)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.strip_prefix("card")
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        })
        .collect();
    cards.sort_by_key(|e| e.file_name());
    cards
        .iter()
        .filter_map(|e| {
            let dev = e.path().join("device");
            let vendor = parse_hex_id(&read(&dev.join("vendor"))?)?;
            let device = parse_hex_id(&read(&dev.join("device"))?)?;
            Some((
                vendor,
                GpuInfo {
                    name: format!("{} GPU (device 0x{device:04x})", vendor_name(vendor)),
                    memory_mb: read(&dev.join("mem_info_vram_total"))
                        .and_then(|s| parse_vram_mb(&s)),
                    cores: kfd_compute_units(kfd_nodes, vendor, device),
                    used_by_ferrum: false,
                },
            ))
        })
        .collect()
}

/// `nvidia-smi` is authoritative for NVIDIA cards (real name and memory), so the
/// sysfs entries for vendor 0x10de are dropped when it answered.
pub fn merge_gpus(smi: Vec<GpuInfo>, sysfs: Vec<(u32, GpuInfo)>) -> Vec<GpuInfo> {
    let have_smi = !smi.is_empty();
    let mut all = smi;
    all.extend(
        sysfs
            .into_iter()
            .filter(|(v, _)| !(have_smi && *v == 0x10de))
            .map(|(_, g)| g),
    );
    all
}

// ── Process helper ───────────────────────────────────────────────────────────

/// Run `program args` and return its stdout if it exits successfully within
/// `timeout`. A missing program, a failure and a timeout all give `None`.
/// (Output is read after exit, which is fine for the few lines `nvidia-smi` prints.)
pub fn run_with_timeout(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flash
    }
    let mut child = cmd.spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut so) = child.stdout.take() {
                    std::io::Read::read_to_string(&mut so, &mut out).ok()?;
                }
                return status.success().then_some(out);
            }
            Ok(None) if start.elapsed() < timeout => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

// ── Detection ────────────────────────────────────────────────────────────────

/// GPUs do not change while the app runs and `nvidia-smi` is slow (and can wake
/// a sleeping card), so they are detected once per process.
fn gpus() -> &'static [GpuInfo] {
    static GPUS: OnceLock<Vec<GpuInfo>> = OnceLock::new();
    GPUS.get_or_init(|| {
        let smi = run_with_timeout(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            Duration::from_secs(2),
        )
        .map(|o| parse_nvidia_smi(&o))
        .unwrap_or_default();
        merge_gpus(
            smi,
            sysfs_gpus(
                Path::new("/sys/class/drm"),
                Path::new("/sys/class/kfd/kfd/topology/nodes"),
            ),
        )
    })
}

fn mib(bytes: u64) -> Option<u64> {
    (bytes > 0).then_some(bytes / (1024 * 1024))
}

/// Detect the current hardware. RAM is read fresh on every call (available RAM
/// changes); the GPU list is cached. May block up to 2 s on the first call.
pub fn detect() -> HardwareInfo {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown CPU".to_string());
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or_else(|_| sys.cpus().len().max(1));
    HardwareInfo {
        cpu_model,
        physical_cores: sys.physical_core_count(),
        logical_cores,
        ferrum_threads: ferrum_core::num_threads(),
        ram_total_mb: mib(sys.total_memory()),
        ram_available_mb: mib(sys.available_memory()),
        gpus: gpus().to_vec(),
    }
}

/// Tauri command: the machine's hardware (see [`HardwareInfo`]).
#[tauri::command]
pub async fn hardware_info() -> Result<HardwareInfo, String> {
    tauri::async_runtime::spawn_blocking(detect)
        .await
        .map_err(|e| format!("task error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn gpu(name: &str, mb: Option<u64>) -> GpuInfo {
        GpuInfo {
            name: name.into(),
            memory_mb: mb,
            cores: None,
            used_by_ferrum: false,
        }
    }

    /// Unique scratch dir under the OS temp dir, removed on drop.
    struct Tmp(PathBuf);
    impl Tmp {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("ferrum_hw_{tag}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Tmp(p)
        }
        fn put(&self, rel: &str, body: &str) {
            let f = self.0.join(rel);
            fs::create_dir_all(f.parent().unwrap()).unwrap();
            fs::write(f, body).unwrap();
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn nvidia_smi_real_output_one_and_many_gpus() {
        // The exact line this project's dev machine prints.
        let one = parse_nvidia_smi("NVIDIA GeForce MX330, 2048, 5\n");
        assert_eq!(one, vec![gpu("NVIDIA GeForce MX330", Some(2048))]);
        let two = parse_nvidia_smi("NVIDIA A100, 40960, 12\r\nNVIDIA T4, 15360, 0\r\n");
        assert_eq!(two.len(), 2);
        assert_eq!(two[1], gpu("NVIDIA T4", Some(15360)));
        assert!(two.iter().all(|g| !g.used_by_ferrum && g.cores.is_none()));
    }

    #[test]
    fn nvidia_smi_tolerates_commas_na_and_garbage() {
        // A comma inside the name stays in the name.
        let g = parse_nvidia_smi("Tesla, Fancy Edition, 8192, 100\n");
        assert_eq!(g, vec![gpu("Tesla, Fancy Edition", Some(8192))]);
        // Memory the driver cannot report.
        assert_eq!(
            parse_nvidia_smi("Some GPU, [N/A], [N/A]\n"),
            vec![gpu("Some GPU", None)]
        );
        // Driver errors, blank input, half lines and non-numeric totals are skipped.
        for bad in [
            "",
            "\n\n",
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
            "name only",
            "x, notanumber, 5",
            ", 100, 5",
            "\u{0}\u{1}garbage,,,",
        ] {
            assert!(parse_nvidia_smi(bad).is_empty(), "{bad:?}");
        }
    }

    #[test]
    fn small_parsers() {
        assert_eq!(parse_hex_id("0x8086\n"), Some(0x8086));
        assert_eq!(parse_hex_id("10de"), Some(0x10de));
        assert_eq!(parse_hex_id("zz"), None);
        assert_eq!(parse_hex_id(""), None);
        assert_eq!(vendor_name(0x10de), "NVIDIA");
        assert!(vendor_name(0xbeef).contains("0xbeef"));
        assert_eq!(parse_vram_mb("8589934592\n"), Some(8192));
        assert_eq!(parse_vram_mb("n/a"), None);
        let props = "cpu_cores_count 0\nsimd_count 64\ncu_count 16\nvendor_id 4098\n";
        assert_eq!(kfd_property(props, "cu_count"), Some(16));
        assert_eq!(kfd_property(props, "vendor_id"), Some(4098));
        assert_eq!(kfd_property(props, "missing"), None);
        assert_eq!(kfd_property("cu_count notanumber", "cu_count"), None);
    }

    #[test]
    fn sysfs_fixture_amd_intel_nvidia_and_connectors() {
        let t = Tmp::new("sysfs");
        // AMD with VRAM, and a KFD node with matching ids (4098 = 0x1002, 26591 = 0x67df).
        t.put("drm/card0/device/vendor", "0x1002\n");
        t.put("drm/card0/device/device", "0x67df\n");
        t.put("drm/card0/device/mem_info_vram_total", "8589934592\n");
        t.put(
            "kfd/0/properties",
            "vendor_id 4098\ndevice_id 26591\ncu_count 36\n",
        );
        t.put(
            "kfd/1/properties",
            "vendor_id 4098\ndevice_id 1\ncu_count 8\n",
        ); // other device
        t.put("kfd/2/properties", "garbage"); // unreadable properties are ignored
                                              // Intel without VRAM file; NVIDIA card; connectors and render nodes are not adapters.
        t.put("drm/card1/device/vendor", "0x8086\n");
        t.put("drm/card1/device/device", "0x9a49\n");
        t.put("drm/card2/device/vendor", "0x10de\n");
        t.put("drm/card2/device/device", "0x1d16\n");
        t.put("drm/card1-eDP-1/device/vendor", "0x8086\n");
        t.put("drm/renderD128/device/vendor", "0x8086\n");
        t.put("drm/card3/device/vendor", "oops\n"); // unparsable: skipped

        let found = sysfs_gpus(&t.0.join("drm"), &t.0.join("kfd"));
        let names: Vec<_> = found.iter().map(|(_, g)| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "AMD GPU (device 0x67df)",
                "Intel GPU (device 0x9a49)",
                "NVIDIA GPU (device 0x1d16)"
            ]
        );
        assert_eq!(found[0].1.memory_mb, Some(8192));
        assert_eq!(found[0].1.cores, Some(36));
        assert_eq!(found[1].1.memory_mb, None);
        assert_eq!(found[1].1.cores, None);
        assert!(found.iter().all(|(_, g)| !g.used_by_ferrum));
    }

    #[test]
    fn sysfs_missing_directories_give_nothing() {
        let t = Tmp::new("missing");
        assert!(sysfs_gpus(&t.0.join("nope"), &t.0.join("nope2")).is_empty());
    }

    #[test]
    fn nvidia_smi_replaces_sysfs_nvidia_but_keeps_other_vendors() {
        let smi = vec![gpu("NVIDIA GeForce MX330", Some(2048))];
        let sys = vec![
            (0x8086, gpu("Intel GPU (device 0x9a49)", None)),
            (0x10de, gpu("NVIDIA GPU (device 0x1d16)", None)),
        ];
        let m = merge_gpus(smi, sys.clone());
        let names: Vec<_> = m.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["NVIDIA GeForce MX330", "Intel GPU (device 0x9a49)"]);
        // Without nvidia-smi the sysfs NVIDIA entry is all we have.
        assert_eq!(merge_gpus(vec![], sys).len(), 2);
    }

    #[test]
    fn process_helper_handles_missing_failing_slow_and_ok_programs() {
        let t = Duration::from_millis(300);
        assert_eq!(run_with_timeout("ferrum-no-such-program-xyz", &[], t), None);
        #[cfg(unix)]
        {
            assert_eq!(
                run_with_timeout("echo", &["hi"], t).as_deref(),
                Some("hi\n")
            );
            assert_eq!(run_with_timeout("false", &[], t), None);
            let start = Instant::now();
            assert_eq!(run_with_timeout("sleep", &["5"], t), None);
            assert!(
                start.elapsed() < Duration::from_secs(3),
                "timeout not honoured"
            );
        }
    }

    #[test]
    fn detect_reports_sane_values_and_never_claims_gpu_use() {
        let hw = detect();
        assert!(hw.logical_cores >= 1);
        assert!(hw.ferrum_threads >= 1);
        assert!(!hw.cpu_model.is_empty());
        if cfg!(target_os = "linux") {
            let total = hw.ram_total_mb.expect("RAM total on Linux");
            let avail = hw.ram_available_mb.expect("RAM available on Linux");
            assert!(total >= 64 && avail <= total, "{total} {avail}");
        }
        assert!(hw.gpus.iter().all(|g| !g.used_by_ferrum));
    }

    #[test]
    fn json_uses_the_camel_case_names_the_ui_reads() {
        let hw = HardwareInfo {
            cpu_model: "X".into(),
            physical_cores: Some(4),
            logical_cores: 8,
            ferrum_threads: 8,
            ram_total_mb: Some(16000),
            ram_available_mb: None,
            gpus: vec![gpu("G", Some(2048))],
        };
        let v = serde_json::to_value(&hw).unwrap();
        for k in [
            "cpuModel",
            "physicalCores",
            "logicalCores",
            "ferrumThreads",
            "ramTotalMb",
            "ramAvailableMb",
            "gpus",
        ] {
            assert!(v.get(k).is_some(), "missing {k}");
        }
        assert_eq!(v["gpus"][0]["usedByFerrum"], false);
        assert_eq!(v["gpus"][0]["memoryMb"], 2048);
        assert!(v["ramAvailableMb"].is_null());
    }
}
