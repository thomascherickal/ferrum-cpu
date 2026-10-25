//! The `--help` text.

pub(crate) fn print_usage() {
    eprintln!(
        "train_transformer — ferrum causal transformer SLM trainer & generator\n\
         (int8 quantization-aware training; weights cached on disk)\n\n\
         USAGE:\n\
         \x20 train_transformer train    <corpus.txt> <model.gguf> [options]\n\
         \x20 train_transformer run      <corpus.txt> <model.gguf> <seed text> [options]\n\
         \x20 train_transformer generate <model.gguf>  <seed text> [options]\n\
         \x20 train_transformer run-gguf <model.gguf> [prompt] [options]\n\
         \x20 train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]\n\
         \x20 train_transformer export-gguf <in.gguf> <out.gguf> [options]\n\
         \x20 train_transformer export-ollama <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]\n\
         \x20 train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]\n\
         \x20 train_transformer eval     <model.gguf>  <heldout.txt> [--json out.json]\n\
         \x20 train_transformer prepare  --input <file|dir> --out <dir> [options]\n\
         \x20 train_transformer info     <model.gguf>\n\n\
         TRAIN / RUN options:\n\
         \x20 --context N  --embed N  --heads N  --blocks N  --hidden N\n\
         \x20 --epochs N   --lr F     --batch N  --vocab N  --seed N\n\
         \x20 --weight_decay F  --dropout F   (AdamW decay + FFN dropout; default 0)\n\
         \x20 --stride N   (window stride; 0 = context [default], 1 = legacy overlap)\n\
         \x20 --tokens N   (token budget, e.g. 5e7; overrides --epochs)\n\
         \x20 --cosine  --warmup N   (warmup + cosine LR decay)   --clip F (grad clip)\n\
         \x20 --tie   (tie LM head to embedding)   --no-qat   (plain fp32 training)\n\
         \x20 --quant int8|int4|f32   (on-disk weight precision; default int8)\n\
         \x20 --tokenizer tok.gguf  (reuse this BPE tokenizer GGUF; trained + saved if missing)\n\
         \x20 --checkpoint ck.fckp  --checkpoint-every N   (save a GGUF checkpoint: weights + optimizer state)\n\
         \x20 --resume ck.fckp      (continue from a GGUF checkpoint)\n\
         \x20 --val F  --patience N (hold out fraction F; keep best epoch; early stop)\n\
         \x20 --threads N  --force    --sample\n\
         \x20 --verbose|-v   (progress line every 10 steps)   --debug   (engine internals)\n\
         \x20 (--vocab 0 = character-level; >=256 = byte-level BPE, default 512)\n\
         \x20 (--threads 0 = auto-detect cores [default]; 1 = serial training)\n\n\
         PREPARE options:\n\
         \x20 --input P (file or dir)  --out DIR  --doc-sep S (default <|endoftext|>)\n\
         \x20 --lossy  --nfkc  --lowercase  --strip-gutenberg  --dedup\n\
         \x20 --min-chars N (1)  --max-chars N (0 = off)\n\
         \x20 --val-frac F (0.05)  --test-frac F (0.05)  --seed N (42)\n\n\
         GENERATE / RUN options:\n\
         \x20 --chars N    --temp F   --gen-seed N   --stream\n\
         \x20 --top-k N    --top-p F  --repetition-penalty F   (sampling knobs; 0/1.0/1.0 = off)\n\
         \x20 --max-tokens N   --stop S   (token budget [default: --chars]; --stop is repeatable,\n\
         \x20     excluded from the output; either flag ignores --chars)\n\
         \x20 (--stream prints the completion live as it is generated)\n\n\
         RUN-GGUF options (import & run a llama/qwen2 GGUF checkpoint):\n\
         \x20 --quant int4|int8|f32  (in-memory precision; default int4)\n\
         \x20 --max N      --temp F  --gen-seed N\n\
         \x20 --top-k N    --top-p F  --rep F  (sampling knobs; 0/1.0/1.0 = off)\n\
         \x20 --ids \"1 2 3\"  (raw prompt token IDs; required if the file has no tokenizer)\n\
         \x20 --raw          (disable automatic chat-template formatting for instruct GGUFs)\n\
         \x20 --force        (load even if the memory estimate exceeds available RAM)\n\
         \x20 --resume ckpt.flck  (overlay fine-tuned weights; forces f32 load)\n\
         \x20 NOTE: only F32/F16/Q8_0/Q8_1/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K GGUFs; on CPU\n\
         \x20 a 1B model decodes at only a few tokens/sec (see Manual.md §13).\n\n\
         FINETUNE-GGUF options (AdamW fine-tune an imported GGUF; f32 masters):\n\
         \x20 --epochs N  --lr F  --batch N  --seq N   (window length; capped at ctx)\n\
         \x20 --warmup N  --clip F  --weight_decay F  --dropout F   (schedule/regularization)\n\
         \x20 --qat          (int8 quantization-aware fine-tuning)\n\
         \x20 --threads N  --seed N  --resume ckpt.flck  --sample\n\
         \x20 writes a .flck checkpoint; apply it with  run-gguf ... --resume out.flck\n\n\
         EXPORT-GGUF options (re-quantize / export an imported GGUF to disk):\n\
         \x20 --quant q8_0|q4_0|q4_1|q8_1|q4_k|q5_k|q6_k|f16|f32  (output precision; default q8_0)\n\
         \x20 --resume ckpt.flck  (apply a fine-tune checkpoint's f32 masters before export)\n\
         \x20 --force        (write even if the memory estimate exceeds available RAM)\n\n\
         If <model.gguf> already exists, train/run load the saved weights from\n\
         disk instead of retraining. Pass --force to retrain from scratch.\n\n\
         EXAMPLES:\n\
         \x20 train_transformer train corpus.txt model.gguf --epochs 20\n\
         \x20 train_transformer train big.txt model.gguf --context 128 --embed 128 --heads 4\n\
         \x20     --blocks 4 --hidden 512 --lr 0.003 --tokens 5e7 --cosine --warmup 200\n\
         \x20     --clip 1 --tie --tokenizer tok.gguf --checkpoint run.fckp --checkpoint-every 500\n\
         \x20 train_transformer run   corpus.txt model.gguf \"Once upon a time\" --chars 300\n\
         \x20 train_transformer generate model.gguf \"Once upon a time\" --temp 0.7"
    );
}
