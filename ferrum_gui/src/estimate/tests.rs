use super::*;
use crate::hardware::GpuInfo;

fn hw(threads: usize, avail_mb: Option<u64>) -> HardwareInfo {
    HardwareInfo {
        cpu_model: "test cpu".into(),
        physical_cores: Some(threads.div_ceil(2)),
        logical_cores: threads,
        ferrum_threads: threads,
        ram_total_mb: Some(16_000),
        ram_available_mb: avail_mb,
        gpus: vec![GpuInfo {
            name: "Big GPU".into(),
            memory_mb: Some(80_000),
            cores: Some(10_000),
            used_by_ferrum: false,
        }],
    }
}

/// A training job at the sizes of a Manual §6.7 row, with a 1,000-byte corpus
/// (so tokenizer time is negligible) and a 1M-token budget.
fn row(
    ctx: usize,
    embed: usize,
    heads: usize,
    blocks: usize,
    hidden: usize,
    vocab: usize,
) -> JobParams {
    JobParams {
        context_len: ctx,
        embed_dim: embed,
        num_heads: heads,
        num_blocks: blocks,
        hidden_dim: hidden,
        vocab_size: vocab,
        batch_size: 16,
        max_tokens: 1_000_000,
        corpus_bytes: Some(1000),
        ..JobParams::default()
    }
}

fn secs(e: &Estimate) -> f64 {
    e.seconds.expect("an estimate")
}

#[test]
fn interp_is_linear_and_clamped() {
    let pts = [(1.0, 10.0), (3.0, 30.0)];
    assert_eq!(interp(&pts, 0.0), 10.0);
    assert_eq!(interp(&pts, 2.0), 20.0);
    assert_eq!(interp(&pts, 9.0), 30.0);
    assert!((interp_log(&[(1.0, 0.0), (100.0, 2.0)], 10.0) - 1.0).abs() < 1e-9);
}

#[test]
fn bytes_per_token_grows_with_vocabulary_and_matches_the_measurements() {
    let mut last = 0.0;
    for v in [0, 256, 300, 512, 1024, 2048, 4096, 50_000] {
        let b = bytes_per_token(v);
        assert!(b >= last, "{v}: {b} < {last}");
        last = b;
    }
    assert_eq!(bytes_per_token(1024), 2.332);
}

#[test]
fn training_speed_reproduces_manual_6_7_within_35_percent() {
    // (params row, Manual tokens/s at 8 threads on the reference machine)
    let rows = [
        (row(64, 64, 4, 2, 256, 512), 16_113.0),
        (row(128, 128, 4, 4, 512, 2048), 3_202.0),
        (row(256, 256, 8, 6, 1024, 4096), 652.0),
        (row(256, 384, 8, 8, 1536, 8192), 268.0),
    ];
    for (p, manual) in rows {
        let e = estimate(JobKind::Train, &p, &hw(8, Some(14_000)), None);
        let rate = 1_000_000.0 / secs(&e);
        let ratio = rate / manual;
        assert!(
            (0.65..1.35).contains(&ratio),
            "{} tokens/s vs Manual {manual}",
            rate
        );
    }
}

#[test]
fn training_ram_is_in_the_range_of_the_measured_peaks() {
    // Manual §6.7 peak RSS. It includes ~130 MB of process, corpus and tokenizer-training
    // memory the estimate (the app's own guard) does not count, so the smallest run is
    // under-estimated and the larger ones are ~1.5x conservative.
    let rows = [
        (row(64, 64, 4, 2, 256, 512), 176.0),
        (row(128, 128, 4, 4, 512, 2048), 297.0),
        (row(256, 256, 8, 6, 1024, 4096), 1400.0),
        (row(256, 384, 8, 8, 1536, 8192), 3000.0),
    ];
    let mut got = vec![];
    for (mut p, measured_mb) in rows {
        p.corpus_bytes = Some(8_000_000);
        let e = estimate(JobKind::Train, &p, &hw(8, Some(14_000)), None);
        got.push((e.peak_ram_mb.unwrap(), measured_mb));
    }
    for (est, measured) in &got {
        assert!((0.25..1.8).contains(&(est / measured)), "{got:?}");
    }
}

#[test]
fn more_cores_are_never_slower_and_beyond_eight_is_flat() {
    let p = row(128, 128, 4, 4, 512, 2048);
    let times: Vec<f64> = [1, 2, 3, 4, 6, 8, 16, 64]
        .iter()
        .map(|&t| secs(&estimate(JobKind::Train, &p, &hw(t, Some(14_000)), None)))
        .collect();
    assert!(times.windows(2).all(|w| w[1] <= w[0] + 1e-9), "{times:?}");
    assert!(
        times[0] > times[5] * 2.0,
        "1 thread should be >2x slower than 8"
    );
    assert_eq!(times[5], times[7], "flat beyond 8");
    // A request for more threads than the machine has does not help.
    let mut greedy = p.clone();
    greedy.threads = 64;
    assert_eq!(
        secs(&estimate(JobKind::Train, &greedy, &hw(4, Some(9000)), None)),
        times[3]
    );
}

#[test]
fn bigger_models_are_not_faster_and_epochs_scale_linearly() {
    let small = row(64, 64, 4, 2, 256, 512);
    let big = row(128, 128, 4, 4, 512, 2048);
    let h = hw(8, Some(14_000));
    assert!(
        secs(&estimate(JobKind::Train, &big, &h, None))
            > secs(&estimate(JobKind::Train, &small, &h, None))
    );
    // Char-level (no tokenizer time): doubling epochs doubles the time exactly.
    let mut one = row(32, 32, 4, 2, 64, 0);
    one.max_tokens = 0;
    one.epochs = 1;
    one.corpus_bytes = Some(100_000);
    let mut three = one.clone();
    three.epochs = 3;
    let (a, b) = (
        secs(&estimate(JobKind::Train, &one, &h, None)),
        secs(&estimate(JobKind::Train, &three, &h, None)),
    );
    assert!((b / a - 3.0).abs() < 1e-9, "{a} {b}");
    // A longer window stride means more passes over the corpus per epoch.
    let mut overlap = one.clone();
    overlap.window_stride = 1;
    assert!(secs(&estimate(JobKind::Train, &overlap, &h, None)) > 20.0 * a);
}

#[test]
fn ram_too_small_is_flagged_and_unknown_ram_is_not() {
    let mut p = row(256, 256, 8, 6, 1024, 4096);
    p.batch_size = 64;
    let small = estimate(JobKind::Train, &p, &hw(8, Some(500)), None);
    assert!(!small.fits_in_ram);
    assert!(
        small.notes.iter().any(|n| n.contains("refused")),
        "{:?}",
        small.notes
    );
    assert!(small.peak_ram_mb.unwrap() > small.ram_limit_mb.unwrap());
    let plenty = estimate(JobKind::Train, &p, &hw(8, Some(64_000)), None);
    assert!(plenty.fits_in_ram);
    let unknown = estimate(JobKind::Train, &p, &hw(8, None), None);
    assert!(unknown.fits_in_ram && unknown.ram_limit_mb.is_none());
    assert!(unknown.notes.iter().any(|n| n.contains("unknown")));
}

#[test]
fn a_faster_benchmark_shortens_the_estimate_and_a_broken_one_is_ignored() {
    let p = row(128, 128, 4, 4, 512, 2048);
    let h = hw(8, Some(14_000));
    let base = secs(&estimate(JobKind::Train, &p, &h, None));
    let same = Throughput {
        gemm_single_gflops: REF_GEMM_SINGLE_GFLOPS,
        mem_bw_gbps: REF_MEM_BW_GBPS,
    };
    assert!((secs(&estimate(JobKind::Train, &p, &h, Some(&same))) - base).abs() < 1e-9);
    let fast = Throughput {
        gemm_single_gflops: 2.0 * REF_GEMM_SINGLE_GFLOPS,
        mem_bw_gbps: REF_MEM_BW_GBPS,
    };
    let e = estimate(JobKind::Train, &p, &h, Some(&fast));
    assert!((secs(&e) / base - 0.5).abs() < 0.01);
    assert!(e.notes[0].contains("Scaled by your Capable benchmark"));
    for bad in [0.0, -1.0, f64::NAN] {
        let b = Throughput {
            gemm_single_gflops: bad,
            mem_bw_gbps: 1.0,
        };
        assert_eq!(secs(&estimate(JobKind::Train, &p, &h, Some(&b))), base);
    }
    // Absurd benchmarks are clamped to 0.25x..4x.
    let silly = Throughput {
        gemm_single_gflops: 1e9,
        mem_bw_gbps: 1e9,
    };
    assert!((secs(&estimate(JobKind::Train, &p, &h, Some(&silly))) / base - 0.25).abs() < 0.01);
}

#[test]
fn training_inputs_that_cannot_run_give_a_reason_not_a_number() {
    let h = hw(8, Some(14_000));
    let good = row(64, 64, 4, 2, 256, 512);
    let mut cases = vec![];
    let mut p = good.clone();
    p.num_heads = 3; // 64 % 3 != 0
    cases.push((p, "divisible"));
    let mut p = good.clone();
    p.embed_dim = 0;
    cases.push((p, "Fill in"));
    let mut p = good.clone();
    p.max_tokens = 0; // and epochs 0
    cases.push((p, "epochs"));
    let mut p = good.clone();
    p.embed_dim = 1 << 20;
    p.hidden_dim = 1 << 22;
    p.num_blocks = 4000;
    cases.push((p, "too large"));
    for (p, word) in cases {
        let e = estimate(JobKind::Train, &p, &h, None);
        assert_eq!(e.seconds, None, "{word}");
        assert!(
            e.notes.iter().any(|n| n.contains(word)),
            "{word}: {:?}",
            e.notes
        );
    }
}

fn gguf(params: u64, embed: u64) -> GgufFacts {
    GgufFacts {
        params,
        embed_params: embed,
        has_output_head: true,
        n_layers: 22,
        context_len: 2048,
        file_bytes: params,
    }
}

#[test]
fn gguf_run_matches_the_manual_for_a_1b_int4_model() {
    let mut p = JobParams {
        gguf: Some(gguf(1_000_000_000, 65_000_000)),
        quant: "int4".into(),
        max_new: 64,
        prompt_chars: 0,
        ..JobParams::default()
    };
    let h = hw(8, Some(14_000));
    // Decode alone: 100 new tokens minus 1 isolates the per-token time.
    let one = secs(&estimate(
        JobKind::GgufRun,
        &JobParams {
            max_new: 1,
            ..p.clone()
        },
        &h,
        None,
    ));
    let many = secs(&estimate(
        JobKind::GgufRun,
        &JobParams {
            max_new: 101,
            ..p.clone()
        },
        &h,
        None,
    ));
    let tok_s = 100.0 / (many - one);
    assert!(
        (4.5..9.5).contains(&tok_s),
        "{tok_s} tokens/s vs Manual 6.68"
    );
    // 512-token prompt: about 25 s of prefill in the Manual (an estimate there too).
    p.prompt_chars = 4 * 504;
    let pre = secs(&estimate(
        JobKind::GgufRun,
        &JobParams {
            max_new: 0,
            ..p.clone()
        },
        &h,
        None,
    )) - 1e9 / 110e6;
    assert!((15.0..40.0).contains(&pre), "{pre} s");
    // int4 needs less RAM than int8 than f32, and the guard is 90% of free RAM.
    let ram = |q: &str| {
        estimate(
            JobKind::GgufRun,
            &JobParams {
                quant: q.into(),
                ..p.clone()
            },
            &h,
            None,
        )
        .peak_ram_mb
        .unwrap()
    };
    assert!(ram("int4") < ram("int8") && ram("int8") < ram("f32"));
    let e = estimate(JobKind::GgufRun, &p, &h, None);
    assert_eq!(e.ram_limit_mb, Some(14_000.0 * 0.9));
    // Fewer cores: slower decode.
    assert!(secs(&estimate(JobKind::GgufRun, &p, &hw(2, Some(14_000)), None)) > secs(&e));
    // No file chosen: no number.
    assert_eq!(
        estimate(JobKind::GgufRun, &JobParams::default(), &h, None).seconds,
        None
    );
}

#[test]
fn finetune_matches_the_manual_and_reports_the_12x_peak() {
    // SmolLM2-135M-like: 135M params, 28M embedding, 3 threads, Manual: ~6 token-passes/s.
    let g = GgufFacts {
        params: 135_000_000,
        embed_params: 28_300_000,
        has_output_head: false,
        n_layers: 30,
        context_len: 2048,
        file_bytes: 270_000_000,
    };
    let p = JobParams {
        gguf: Some(g),
        corpus_bytes: Some(4 * 63),
        epochs: 1,
        batch_size: 8,
        seq_len: 64,
        threads: 3,
        ..JobParams::default()
    };
    let h = hw(8, Some(30_000));
    let e = estimate(JobKind::GgufFinetune, &p, &h, None);
    // One window of 64 tokens: 64 passes.
    let train = secs(&e) - 135e6 / 110e6;
    let passes_per_s = 64.0 / train;
    assert!((3.0..12.0).contains(&passes_per_s), "{passes_per_s}");
    assert_eq!(e.peak_ram_mb, Some(135e6 * 4.0 * 12.0 / 1048576.0));
    assert!(!estimate(JobKind::GgufFinetune, &p, &hw(8, Some(4000)), None).fits_in_ram);
    // More epochs, proportionally longer.
    let two = secs(&estimate(
        JobKind::GgufFinetune,
        &JobParams {
            epochs: 2,
            ..p.clone()
        },
        &h,
        None,
    )) - 135e6 / 110e6;
    assert!((two / train - 2.0).abs() < 1e-6);
    // A corpus shorter than the sequence cannot be fine-tuned.
    assert_eq!(
        estimate(
            JobKind::GgufFinetune,
            &JobParams {
                corpus_bytes: Some(10),
                ..p
            },
            &h,
            None
        )
        .seconds,
        None
    );
}

#[test]
fn export_ram_mirrors_the_apps_own_guard() {
    let g = gguf(10_000_000, 1_000_000);
    let base = JobParams {
        gguf: Some(g),
        quant: "q8_0".into(),
        ..JobParams::default()
    };
    let h = hw(8, Some(14_000));
    let peak = |q: &str, resume: &str| {
        let p = JobParams {
            quant: q.into(),
            resume_path: resume.into(),
            ..base.clone()
        };
        estimate(JobKind::GgufExport, &p, &h, None)
            .peak_ram_mb
            .unwrap()
            * 1048576.0
    };
    let model = 40_000_000.0;
    assert_eq!(peak("f32", ""), model + model);
    assert_eq!(peak("f16", ""), model + model / 2.0);
    assert_eq!(peak("q4_k", ""), model + (40_000_000usize / 3) as f64);
    assert_eq!(peak("q8_0", "tuned.flck"), model * 4.0); // 4x for gradients + Adam moments
    assert_eq!(
        estimate(JobKind::GgufExport, &base, &h, None)
            .seconds
            .unwrap(),
        10e6 / 70e6
    );
}

#[test]
fn generate_and_evaluate_use_the_model_facts() {
    let h = hw(8, Some(14_000));
    let slm = |matmul: u64| SlmFacts {
        params: matmul + 100_000,
        matmul_params: matmul,
        vocab: 1024,
        file_bytes: 2_000_000,
    };
    let gen = |m: u64, chars: usize| {
        secs(&estimate(
            JobKind::Generate,
            &JobParams {
                slm: Some(slm(m)),
                num_chars: chars,
                ..JobParams::default()
            },
            &h,
            None,
        ))
    };
    assert!(gen(400_000, 2000) > gen(400_000, 200));
    assert!(gen(4_000_000, 200) > gen(400_000, 200));
    // Generation does not speed up with cores.
    let p = JobParams {
        slm: Some(slm(400_000)),
        num_chars: 600,
        ..JobParams::default()
    };
    assert_eq!(
        secs(&estimate(JobKind::Generate, &p, &hw(2, Some(14_000)), None)),
        secs(&estimate(JobKind::Generate, &p, &h, None))
    );
    // A token budget replaces the character budget (the Generate tab's "Max tokens").
    let tokens = |max_new: usize| {
        secs(&estimate(
            JobKind::Generate,
            &JobParams {
                slm: Some(slm(400_000)),
                num_chars: 200,
                max_new,
                ..JobParams::default()
            },
            &h,
            None,
        ))
    };
    assert!(
        tokens(2000) > tokens(0),
        "2000 tokens cost more than 200 chars"
    );
    assert!(tokens(10) < tokens(0), "10 tokens cost less than 200 chars");
    // ~ Manual §13.2: 0.26-0.29 ms/token for the 0.5M model (+ load): 600 chars is ~257 tokens.
    assert!(
        (0.03..0.4).contains(&(gen(396_960, 600) - 0.05 - 2e6 / 200e6)),
        "{}",
        gen(396_960, 600)
    );
    let ev = |bytes: u64, t: usize| {
        secs(&estimate(
            JobKind::Evaluate,
            &JobParams {
                slm: Some(slm(400_000)),
                text_bytes: Some(bytes),
                ..JobParams::default()
            },
            &hw(t, Some(14_000)),
            None,
        ))
    };
    assert!(ev(200_000, 8) > ev(100_000, 8));
    assert!(ev(100_000, 1) > ev(100_000, 8));
    // Pasted text counts when there is no file.
    let pasted = JobParams {
        slm: Some(slm(400_000)),
        text_chars: 100_000,
        ..JobParams::default()
    };
    assert!((secs(&estimate(JobKind::Evaluate, &pasted, &h, None)) - ev(100_000, 8)).abs() < 1e-9);
}

#[test]
fn evaluating_an_imported_gguf_counts_load_scoring_precision_and_the_test_file() {
    let h = hw(8, Some(14_000));
    let eval = |quant: &str, resume: &str, test: u64| {
        estimate(
            JobKind::Evaluate,
            &JobParams {
                gguf: Some(gguf(135_000_000, 28_000_000)),
                text_bytes: Some(1_000_000),
                test_bytes: Some(test),
                quant: quant.into(),
                resume_path: resume.into(),
                resume_bytes: (!resume.is_empty()).then_some(500_000_000),
                ..JobParams::default()
            },
            &h,
            None,
        )
    };
    let f32e = eval("f32", "", 0);
    assert!(f32e.seconds.is_some_and(|s| s > 0.0), "{f32e:?}");
    assert!(
        f32e.notes.iter().any(|n| n.contains("Imported GGUF")),
        "{:?}",
        f32e.notes
    );
    // int8 needs less RAM and is faster than f32; a test file and a checkpoint cost more.
    let i8e = eval("int8", "", 0);
    assert!(i8e.peak_ram_mb < f32e.peak_ram_mb && i8e.seconds < f32e.seconds);
    assert!(eval("f32", "", 1_000_000).seconds > f32e.seconds);
    let ck = eval("int8", "/c.flck", 0);
    assert!(
        ck.peak_ram_mb > f32e.peak_ram_mb,
        "a checkpoint forces f32 and adds its bytes"
    );
    // Without a model it still falls back to the Ferrum-model default instead of failing.
    let none = estimate(JobKind::Evaluate, &JobParams::default(), &h, None);
    assert!(none.seconds.is_some());
}

#[test]
fn small_jobs_download_and_instant_kinds() {
    let h = hw(8, Some(14_000));
    let p = JobParams {
        max_mb: 100.0,
        ..JobParams::default()
    };
    let d = estimate(JobKind::Download, &p, &h, None);
    assert_eq!(d.seconds, Some(20.0));
    assert!(d.low_seconds.unwrap() < 20.0 && d.high_seconds.unwrap() > 20.0);
    assert!(d
        .notes
        .iter()
        .any(|n| n.contains("Depends on your connection")));
    assert!(estimate(JobKind::Download, &JobParams::default(), &h, None)
        .notes
        .iter()
        .any(|n| n.contains("50 MB")));
    let i = estimate(JobKind::Instant, &p, &h, None);
    assert_eq!(
        (i.seconds, i.peak_ram_mb, i.notes.len(), i.uses_gpu),
        (Some(0.0), None, 0, false)
    );
    let c = estimate(JobKind::Capability, &p, &h, None);
    assert_eq!(c.seconds, Some(1.5));
    let clean = |n| {
        secs(&estimate(
            JobKind::CleanText,
            &JobParams {
                text_chars: n,
                ..p.clone()
            },
            &h,
            None,
        ))
    };
    assert!(clean(100_000_000) > clean(1000) && clean(1000) < 0.1);
}

#[test]
fn every_kind_returns_an_estimate_and_a_gpu_never_changes_it() {
    let kinds = [
        JobKind::Instant,
        JobKind::Download,
        JobKind::LoadFile,
        JobKind::CleanText,
        JobKind::Train,
        JobKind::Generate,
        JobKind::Evaluate,
        JobKind::ModelInfo,
        JobKind::GgufInfo,
        JobKind::GgufRun,
        JobKind::GgufFinetune,
        JobKind::GgufExport,
        JobKind::Capability,
    ];
    let mut nogpu = hw(8, Some(14_000));
    nogpu.gpus.clear();
    let with = hw(8, Some(14_000));
    for k in kinds {
        let p = JobParams::default();
        let (a, b) = (estimate(k, &p, &with, None), estimate(k, &p, &nogpu, None));
        assert_eq!(a, b, "{k:?}: a GPU must not change an estimate");
        assert!(!a.uses_gpu && a.kind == k);
        if k != JobKind::Instant {
            assert!(
                a.notes.iter().any(|n| n.contains("not used by Ferrum")),
                "{k:?}"
            );
        }
        if let (Some(lo), Some(s), Some(hi)) = (a.low_seconds, a.seconds, a.high_seconds) {
            assert!(lo <= s && s <= hi, "{k:?}");
        }
    }
}

#[test]
fn json_in_and_out_match_what_the_ui_sends_and_reads() {
    let k: JobKind = serde_json::from_str("\"gguf_finetune\"").unwrap();
    assert_eq!(k, JobKind::GgufFinetune);
    assert!(serde_json::from_str::<JobKind>("\"nonsense\"").is_err());
    // Missing fields default; unknown fields are ignored.
    let p: JobParams =
        serde_json::from_str(r#"{"contextLen":16,"embedDim":32,"maxTokens":50000000,"extra":1}"#)
            .unwrap();
    assert_eq!(
        (p.context_len, p.embed_dim, p.max_tokens, p.epochs),
        (16, 32, 50_000_000, 0)
    );
    let e = estimate(
        JobKind::Train,
        &row(16, 32, 4, 2, 64, 512),
        &hw(8, Some(9000)),
        None,
    );
    let v = serde_json::to_value(&e).unwrap();
    for key in [
        "kind",
        "seconds",
        "lowSeconds",
        "highSeconds",
        "peakRamMb",
        "ramLimitMb",
        "ramAvailableMb",
        "fitsInRam",
        "threadsUsed",
        "usesGpu",
        "notes",
    ] {
        assert!(v.get(key).is_some(), "missing {key}");
    }
    assert_eq!(v["kind"], "train");
}
