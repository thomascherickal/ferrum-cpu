// Help texts for every field and button of the UI. This file is DATA: edit the
// strings freely, no logic depends on their wording. help.test.js checks that
// every id in index.html has an entry and that the texts are long enough.
//
//   field entry : { title, what, how, beginner }    (input / textarea / select)
//   button entry: { title, does, kind, how?, beginner? }
//
// `kind` is the estimate job kind (the snake_case names of JobKind in
// src/estimate.rs); the tooltip adds a live time/RAM estimate for it.
// Plain text only: the tooltip sets textContent, so "<" and "&" are safe but
// are kept out of the data anyway.
"use strict";
(function (global) {
  const KINDS = [
    "instant", "download", "load_file", "clean_text", "train", "generate", "evaluate",
    "model_info", "gguf_info", "gguf_run", "gguf_finetune", "gguf_export", "capability", "prepare", "runtime_export", "tokenize", "serve",
  ];

  // A "Browse..." button next to a path field.
  const browse = (what, mode) => ({
    title: "Browse for " + what,
    does: mode === "save"
      ? "Opens the system 'save as' dialog and puts the file path you pick into the " + what + " field, so you do not have to type it."
      : mode === "add" || mode === "adddir"
        ? "Opens the system " + (mode === "add" ? "file" : "folder") + " picker and adds the path you choose as a new line in the input list, so you can build up several inputs without typing."
      : mode === "dir"
        ? "Opens the system folder picker and puts the folder you choose into the " + what + " field, so you do not have to type it."
        : "Opens the system file picker and puts the path of the file you choose into the " + what + " field, so you do not have to type it.",
    kind: "instant",
    beginner: "A path is the address of a file on your computer, like /home/you/data/story.txt. Browsing just fills in that address for you; nothing is read or written until you press the action button.",
  });
  const tab = (name, what, how) => ({
    title: name + " tab",
    does: "Shows the " + name + " page: " + what,
    how,
    kind: "instant",
  });

  const HELP = {
    // ── Datasets ──────────────────────────────────────────────────────────
    dsDestDir: {
      title: "Download folder",
      what: "The folder on your computer where the catalog, HuggingFace and Kaggle downloads are saved. It is created if it does not exist.",
      how: "Type a folder path or use Browse. After a download finishes, the 'or local file' field below is filled with the new file so you can load it with one click. Downloads go to a temporary .part file first and are renamed only when complete, so a broken connection never leaves a half file.",
      beginner: "A dataset is just a big text file the model learns from. You need somewhere to keep it; pick any folder you can write to, for example a 'datasets' folder in your home directory.",
    },
    dsHfRepo: {
      title: "HuggingFace repo",
      what: "The name of a dataset repository on huggingface.co, written owner/name, for example roneneldan/TinyStories.",
      how: "Together with the File field it forms the download address https://huggingface.co/datasets/OWNER/NAME/resolve/main/FILE. Only the single file you name is downloaded (up to 512 MiB), never the whole repository.",
      beginner: "HuggingFace is a website that hosts free datasets and models. Open a dataset's page, copy its name from the top (owner/name), then look in its 'Files' tab for the exact file name to type in the next box.",
    },
    dsHfFile: {
      title: "HuggingFace file",
      what: "The exact name of one file inside the HuggingFace repository, for example TinyStories-valid.txt.",
      how: "It must match the name in the repo's Files tab, including capital letters. Plain text files (.txt) can go straight to Clean; other formats (CSV, JSON, parquet) are downloaded as they are and need converting to text first.",
      beginner: "Datasets are often split into several files, such as a training file and a smaller validation file. Pick a small one first (a few MB) to try the pipeline quickly on a CPU.",
    },
    dsHfToken: {
      title: "HuggingFace token (gated repos)",
      what: "An optional access token for repositories that ask you to accept terms before downloading. Leave it empty for public datasets.",
      how: "If empty, Ferrum uses the HF_TOKEN environment variable when it is set. The token is sent to huggingface.co as an authorization header for that one download and is not saved by Ferrum.",
      beginner: "A token is like a password for an app. You create one in your HuggingFace account settings. Treat it as secret: do not share screenshots that show it.",
    },
    dsKgRepo: {
      title: "Kaggle owner/slug",
      what: "The Kaggle dataset to fetch from, written owner/dataset-name (the last two parts of its Kaggle web address).",
      how: "Kaggle needs credentials: set the KAGGLE_USERNAME and KAGGLE_KEY environment variables, or place kaggle.json in ~/.kaggle/. Only the single file named in the next field is downloaded.",
      beginner: "Kaggle is a data-science site with thousands of datasets. You need a free account and an API key (Account page, 'Create New Token') before downloads work.",
    },
    dsKgFile: {
      title: "Kaggle file",
      what: "The exact file name inside the Kaggle dataset, for example data.csv.",
      how: "Copy it from the dataset's Data tab. The file is saved into the download folder unchanged; a CSV is not text a language model can use directly, so pick a text file when you can.",
      beginner: "Language models learn from plain text. If the file is a table (CSV), you would first extract the text column with another tool.",
    },
    dsUrl: {
      title: "Source URL",
      what: "A web address (http:// or https://) of a plain text file to download, such as a Project Gutenberg book.",
      how: "Press Download to fetch it. The text is decoded as UTF-8 (unreadable bytes are replaced) and shown in the Raw text box. Connections time out after 15 seconds to connect and 30 seconds without data, and the file is cut off at the Max MB limit next to it.",
      beginner: "Project Gutenberg offers thousands of free books as .txt files, a classic first training text. Make sure the address ends in .txt and opens as plain text in your browser.",
    },
    dsMaxMb: {
      title: "Max MB",
      what: "The largest download in megabytes that the Download button will accept. The default is 8.",
      how: "Anything beyond this limit is simply not read, so a big file is truncated rather than rejected. A larger limit means longer downloads, more memory while cleaning, and longer training later.",
      beginner: "One megabyte is roughly a million characters, about a thousand pages of text. A few MB is plenty for a first small model; more text usually gives a better model but takes proportionally longer to train.",
    },
    dsFile: {
      title: "Local file",
      what: "The path of a text file already on your computer that you want to use as training text.",
      how: "Press Load file to read it (up to 16 MiB) into the Raw text box. Devices and folders are refused; the text is decoded as UTF-8 and bad bytes are replaced. This field is filled automatically after a catalog, HuggingFace or Kaggle download.",
      beginner: "If you already have a text file, use this instead of downloading. Any plain text works: stories, documentation, chat logs, code. The model will imitate whatever you give it.",
    },
    dsRaw: {
      title: "Raw text",
      what: "The unprocessed text. It fills automatically from a download or a loaded file, and you can also paste or type text straight into it.",
      how: "Clean and preview reads this box, applies the cleaning options and shows the result below. Your edits here are used as they are, so you can remove junk by hand first.",
      beginner: "Raw means 'not cleaned yet'. Real-world text has page headers, odd symbols and blank lines that waste the model's limited capacity, which is why a cleaning step exists.",
    },
    optGutenberg: {
      title: "Strip Project Gutenberg boilerplate",
      what: "Keeps only the book text between the START and END markers that Project Gutenberg puts in its files, dropping the legal header and footer.",
      how: "If the markers are not found the text passes through unchanged. It also joins several books in one file. Turn it off for any text that is not from Gutenberg; it does no harm there but is pointless.",
      beginner: "Gutenberg files start with pages of license text. Left in, a small model would learn to write license text, so it is better removed.",
    },
    optCollapse: {
      title: "Collapse blank lines and trailing spaces",
      what: "Trims spaces at the end of lines and squeezes long runs of blank lines down to a small number.",
      how: "Makes the text more regular so the model does not spend capacity predicting layout noise. Line breaks inside paragraphs are kept.",
      beginner: "To a model, every space and newline is a character it has to learn to predict. Fewer pointless ones means more effort goes to real words.",
    },
    optPunct: {
      title: "Normalize smart punctuation to ASCII",
      what: "Replaces curly quotes, long dashes, non-breaking spaces, bullets and similar look-alikes with plain keyboard characters.",
      how: "Shrinks the number of distinct characters. With a small vocabulary this matters, because every rare symbol takes a slot. Leave it on for English text; turn it off if the exact typography matters to you.",
      beginner: "A curly quote and a straight quote look almost the same to you but are different symbols to the computer. Merging them makes the model's job easier.",
    },
    optControl: {
      title: "Strip control characters",
      what: "Removes invisible control characters and zero-width marks (such as a byte-order mark) but keeps normal newlines and tabs.",
      how: "These characters are invisible to you but would still be learned as tokens. Removing them is almost always right; carriage returns are always removed whatever you choose.",
      beginner: "Some text carries hidden characters from copy-and-paste or old file formats. They never help a language model.",
    },
    optLower: {
      title: "Lowercase",
      what: "Converts all text to lowercase before training.",
      how: "Shrinks the vocabulary (no separate 'The' and 'the'), which helps tiny models and character-level models. The cost is that generated text will be all lowercase.",
      beginner: "Vocabulary means the set of distinct pieces the model knows. Lowercasing roughly halves the letters it must learn, a useful trick when the model and data are small.",
    },
    optMaxChars: {
      title: "Max chars (0 = all)",
      what: "Keeps only the first N characters of the cleaned text. 0 keeps everything.",
      how: "Useful for a quick trial run: for example 200000 gives a corpus that trains in minutes. The cut happens last, after the other options.",
      beginner: "More data is better but slower. Start with a slice (say 200,000 characters) to check that everything works, then rerun with 0 to use all of it.",
    },
    dsPreview: {
      title: "Cleaned preview",
      what: "A read-only view of the first 4,000 characters of the cleaned text, so you can check the result by eye.",
      how: "It only shows the preview; Save corpus writes the whole cleaned text, not just what you see here. The statistics next to the Clean button show the full size.",
      beginner: "Always glance at this before training. If it still contains junk (menus, numbers, code you did not want), the model will learn it too: garbage in, garbage out.",
    },
    dsOut: {
      title: "Save to",
      what: "The file path where the cleaned corpus will be written, for example /home/you/data/clean_corpus.txt.",
      how: "Missing folders are created. An existing file at that path is overwritten. After saving, the Train tab's Corpus field is filled in for you if it was empty.",
      beginner: "Think of it as 'Save As'. The saved text file is what you point the Train tab at.",
    },

    // ── Train ─────────────────────────────────────────────────────────────
    trCorpus: {
      title: "Corpus",
      what: "The path of the training text file (the corpus), or of a token file made by Tokenize corpus below. The model learns to continue text like this.",
      how: "A text file must be non-empty and at least a little longer than the Context length in tokens, or training refuses to start. A token file is recognised by its content, needs the Tokenizer file field to name the tokenizer it was made with, and is read from disk window by window, so memory does not grow with its size. Use the Prepare tab to clean and split a corpus. The size drives the time estimate.",
      beginner: "A corpus is the collection of text the model studies. Everything the model will ever know comes from it, so its quality and size matter more than any setting below.",
    },
    trModel: {
      title: "Model output",
      what: "Where the trained model is saved, as a .gguf file, for example /home/you/models/story.gguf.",
      how: "Training writes the file atomically at the end (a temporary file, then a rename), so an old model at the same path stays intact if training fails. The same file runs in llama.cpp, Ollama and LM Studio.",
      beginner: "GGUF is a common file format for language models. One .gguf file contains the whole model: its numbers and its tokenizer.",
    },
    trContext: {
      title: "Context (tokens)",
      what: "The context window: how many previous tokens the model can look at when it predicts the next one. Default 16.",
      how: "Training chops the corpus into windows of this length. Longer context lets the model use more history but costs more memory (attention grows with context) and time. Typical small values are 64 to 256. The model cannot use more history than this when generating.",
      beginner: "A token is a small piece of text, roughly a short word or a part of a word. A context of 64 tokens is about 50 words. The model is like a reader who only remembers the last page.",
    },
    trEmbed: {
      title: "Embed dim",
      what: "The embedding width: how many numbers describe each token inside the model (the hidden size). Default 32.",
      how: "It must be divisible by Heads. Bigger means a more capable model but roughly four times the parameters when you double it, and slower, hungrier training. Examples: 64 for a toy, 128 to 256 for a small model.",
      beginner: "An embedding turns each token into a list of numbers that captures its meaning. A longer list can hold more nuance, like describing a person with ten facts rather than two.",
    },
    trHeads: {
      title: "Heads",
      what: "The number of attention heads in each block. Default 4.",
      how: "Each head looks at the context in its own way; the embedding is split evenly between them, so Embed dim must be divisible by Heads (the Train button checks this). 4 to 8 is typical for small models. It changes the structure, not the parameter count.",
      beginner: "Attention is how the model decides which earlier words matter for the next word. Multiple heads let it track several things at once, such as grammar and topic.",
    },
    trBlocks: {
      title: "Blocks",
      what: "The number of stacked Transformer blocks (layers). Default 2, at least 1.",
      how: "Each block adds about 12 times Embed dim squared parameters when Hidden is 4 times Embed dim. More blocks learn more abstract patterns but multiply training time and memory almost linearly. 2 to 6 suits CPU training.",
      beginner: "A block is one round of 'look at the context, then think about it'. Stacking blocks is like reading a sentence again and again, each pass understanding a bit more.",
    },
    trHidden: {
      title: "Hidden",
      what: "The width of the feed-forward layer inside each block. Default 64.",
      how: "The usual choice is 4 times Embed dim (so 128 for Embed 32). Wider hidden layers add parameters and compute in proportion. It must be at least 1.",
      beginner: "After attention, each block runs every token through a small neural network. This setting is how many neurons that network has: its 'thinking space'.",
    },
    trEpochs: {
      title: "Epochs",
      what: "How many full passes over the corpus training makes. Default 100, at least 1. Ignored when a Token budget is set.",
      how: "Time grows in direct proportion to epochs. With the default stride each epoch sees every token once. Watch the loss chart: when it stops falling, or when the held-out perplexity on the Evaluate tab gets worse, more epochs only memorize.",
      beginner: "One epoch means the model has read the whole text once. Like studying for an exam, several readings help, but too many just means memorizing the page instead of learning the language.",
    },
    trLr: {
      title: "LR (learning rate)",
      what: "How big a step the optimizer (Adam) takes when it corrects the model after each batch. Default 0.01; must be above 0.",
      how: "Small models tolerate 0.003 to 0.01. Above about 5 million parameters start nearer 0.001 to 0.003, with Cosine LR and some Warmup steps. Too high and the loss jumps around or the run aborts as diverged; too low and learning crawls.",
      beginner: "Imagine walking downhill in fog to find the lowest point. The learning rate is your stride length: huge strides overshoot the valley, tiny strides take forever.",
    },
    trBatch: {
      title: "Batch",
      what: "How many text windows are processed together for each update of the model. Default 16, at least 1.",
      how: "Bigger batches give smoother, more reliable updates and use the CPU better, but need more RAM (activations scale with batch times context) and make fewer updates per epoch. Also caps the number of useful training threads.",
      beginner: "Instead of learning from one sentence at a time, the model studies a stack of them and averages what it learned. It is like a teacher marking 16 essays before giving feedback.",
    },
    trVocab: {
      title: "Vocab (0 or 256 and up)",
      what: "The vocabulary size, the number of distinct tokens. 0 means character-level (one token per distinct character); 256 or more means byte-level BPE of that size. Default 512.",
      how: "Values from 1 to 255 are rejected. BPE packs more text into each context window and handles any language; try 512 to 4096. A small corpus may supply fewer merges than requested and the model then simply has a smaller vocabulary. A larger vocabulary adds Embed dim parameters per token.",
      beginner: "A tokenizer chops text into pieces. BPE (byte pair encoding) learns common pieces from your data, like 'ing' or ' the', so frequent words become one token and rare words are spelled out from smaller parts.",
    },
    trThreads: {
      title: "Threads (0 = auto)",
      what: "The number of data-parallel training workers. 0 picks Ferrum's thread count (the number of CPU cores, or FERRUM_NUM_THREADS).",
      how: "Each worker handles part of every batch, so it cannot help beyond the Batch size. Speed-up is real but sub-linear: on a 4-core laptop 8 threads gave 2.4 times, not 8. A different thread count trains slightly different weights, so fix it if you need to reproduce a run exactly.",
      beginner: "A thread is one worker the CPU can run at the same time. More workers finish sooner, up to the point where the cores are all busy.",
    },
    trSeed: {
      title: "Seed",
      what: "The starting number of the random generator that initialises the weights and shuffles the data. Default 1337.",
      how: "The same seed, settings, corpus and thread count give a byte-identical model every time. Change the seed to see how much results vary by luck.",
      beginner: "Computers make 'random' numbers by a fixed recipe starting from a seed. Keeping the seed makes experiments repeatable, which is how you compare two settings fairly.",
    },
    trStride: {
      title: "Stride (0 = context)",
      what: "The number of tokens between the starts of consecutive training windows. 0 means 'equal to Context', so every token is seen once per epoch.",
      how: "A stride of 1 is the legacy full overlap: about Context times more work per epoch for little gain. Values between 1 and Context overlap windows partially and multiply the epoch time accordingly.",
      beginner: "Training reads the text in chunks. Stride says how far to slide before cutting the next chunk. Sliding by a whole chunk covers the text once; sliding by one token covers it over and over.",
    },
    trTokens: {
      title: "Token budget (0 = off)",
      what: "Stop after this many training tokens, for example 5e7 for fifty million. 0 means use Epochs instead. Overrides Epochs when set.",
      how: "A rule of thumb for a well-trained model is about 20 tokens per parameter, so a 1.5 million parameter model wants about 3e7. Training can stop in the middle of an epoch. The progress bar is not filled in this mode because the epoch count is unknown.",
      beginner: "A token budget says how much reading to do rather than how many times to read the book. It is the fairer way to compare different corpora and sizes.",
    },
    trWd: {
      title: "Weight decay",
      what: "A gentle pull that shrinks all weights toward zero at every step (AdamW). Default 0 (off).",
      how: "Try 0.01 to 0.1 when the held-out perplexity is clearly worse than the training loss suggests, the sign of memorization. Too much stops the model from learning.",
      beginner: "Weights are the model's numbers. Keeping them small acts like a rule 'do not rely on any one fact too much', which makes the model generalise better to new text.",
    },
    trDropout: {
      title: "Dropout",
      what: "The fraction of the feed-forward neurons randomly switched off during each training step, from 0 to 0.95. Default 0 (off).",
      how: "0.1 is a common value when the model memorizes a small corpus. It only applies while training; the saved model uses all neurons. High values slow learning.",
      beginner: "Randomly muting parts of the brain while studying forces the rest to learn robustly, so no single neuron becomes a crutch. It is a common cure for memorizing.",
    },
    trClip: {
      title: "Grad clip (0 = off)",
      what: "Caps the overall size of each update (the gradient norm). 0 turns clipping off.",
      how: "A value of 1 is cheap insurance against rare exploding updates on longer runs; it rarely hurts. Updates smaller than the cap are untouched.",
      beginner: "Occasionally the model gets a huge correction signal that would wreck what it learned. Clipping limits the size of any single correction so training stays stable.",
    },
    trWarmup: {
      title: "Warmup steps",
      what: "The number of optimizer steps over which the learning rate rises linearly from zero to its full value. Default 0.",
      how: "Only used together with Cosine LR. For models above a few million parameters 100 to 500 steps help avoid early instability. One step handles one batch.",
      beginner: "Like warming up before a run: starting with tiny updates lets the model find its footing before it takes full-size steps.",
    },
    trVal: {
      title: "Val fraction (0 = off)",
      what: "The share of the corpus held out for validation, from 0 to 0.5. 0 means no hold-out.",
      how: "The last part of the text is held out; after every epoch the model is scored on it and the weights with the best validation loss are the ones saved. A tokenizer trained in the same run only sees the training part. A value like 0.05 costs 5 percent of your data.",
      beginner: "Validation means testing on pages the model did not study. It tells you whether the model really learned the language or just memorized the training text.",
    },
    trPatience: {
      title: "Patience",
      what: "With a Val fraction: stop training after this many epochs in a row without an improvement on the held-out text. 0 means never stop early.",
      how: "Saves time on long runs: a patience of 5 to 10 is typical. The best epoch's weights are kept either way. It does nothing without a validation fraction.",
      beginner: "'Early stopping' means quitting once more studying no longer helps. Patience is how many disappointing rounds you tolerate before you give up.",
    },
    trTokenizer: {
      title: "Tokenizer file",
      what: "Optional path of a saved BPE tokenizer (a GGUF file) to reuse. If the file does not exist yet, the tokenizer is trained and saved there.",
      how: "Reusing one keeps the vocabulary stable between experiments and resumes and skips tokenizer training. Not used for character-level models (Vocab 0). It must not be the same file as the model output. Tokenize corpus below needs an existing one, and a token file only trains with the tokenizer it was made with.",
      beginner: "The tokenizer is the dictionary that converts text to numbers. If you train several models on the same text, sharing one tokenizer keeps them comparable.",
    },
    trCkpt: {
      title: "Checkpoint file",
      what: "Optional path where the full training state (weights, optimizer memory and random state) is saved so a run can be continued later.",
      how: "Written every 'Checkpoint every' steps and always at the end. Combine with the Resume box to continue after an interruption. It must be a different file from the model output.",
      beginner: "A checkpoint is a save game. If the power goes out after three hours of training, you restart from the last save instead of from scratch.",
    },
    trCkptEvery: {
      title: "Checkpoint every",
      what: "Save the checkpoint every N optimizer steps. 0 means only once at the end.",
      how: "One step processes one batch. Saving more often costs a little time per save but loses less work after a crash. A few hundred to a few thousand steps is typical for long runs. Needs a Checkpoint file.",
      beginner: "How often the game autosaves. Too often wastes time on saving, too rarely risks losing progress.",
    },
    trCosine: {
      title: "Cosine LR (with warmup)",
      what: "Uses a learning-rate schedule: a linear warmup, then a smooth cosine-shaped decay to zero over the whole run.",
      how: "The LR field becomes the peak rate. The schedule needs to know the total number of steps, which comes from Epochs or the Token budget. It usually gives a better final loss than a constant rate on longer runs.",
      beginner: "Big steps early, tiny steps at the end, like first sketching a drawing and then adding fine detail.",
    },
    trTie: {
      title: "Tie embedding + head",
      what: "Makes the output layer reuse the token embedding table instead of having its own.",
      how: "Removes Vocab times Embed dim parameters and makes the file smaller by that much. It often helps small models. The saved file then has no output.weight tensor.",
      beginner: "The model uses a table to turn tokens into numbers on the way in and another to turn numbers back into tokens on the way out. Tying means using the same table both ways, saving space.",
    },
    trQat: {
      title: "Int8 QAT",
      what: "Quantization-aware training: trains while pretending weights are stored with 8-bit precision (the Q8_0 format).",
      how: "Costs a little speed. Keep it on if you will ship the int8 model, because the model learns to be accurate even after rounding. Turn it off for plain 32-bit float training. There is no int4 QAT.",
      beginner: "Quantization stores each number with fewer bits, shrinking the file by about four times. Training with the rounding in the loop makes the small version nearly as good as the big one.",
    },
    trResume: {
      title: "Resume from checkpoint",
      what: "Continue a previous run from the Checkpoint file instead of starting from scratch.",
      how: "Needs an existing checkpoint file (the Train button says so otherwise). The run keeps the epoch and step counts, the optimizer's memory and the random state, and keeps writing to that checkpoint. Keep the other settings the same.",
      beginner: "Loads your save game and carries on where you stopped, for example after an interruption or to train for more epochs.",
    },
    trForce: {
      title: "I know, try anyway",
      what: "Appears only when the pre-flight check says the run needs more memory than the engine allows. Ticking it lets Start run anyway and skips the engine's own refusal.",
      how: "The engine normally refuses a run that needs more than three quarters of your free RAM (and never more than 16 GB), because a run that does not fit makes the computer swap to disk and crawl, or the system kills the program. Ticking this removes that safety net. Lowering batch, context, embed dim or blocks is the better fix.",
      beginner: "RAM is the computer's working memory. If a job needs more than there is, everything slows to a crawl. The pre-flight check warns you before you start; this box says you accept the risk.",
    },
    trVerbose: {
      title: "Verbose",
      what: "Streams the engine's internal diagnostic lines into the Engine log panel at the bottom of the window.",
      how: "This is the engine's internal trace, so it can be very chatty and slightly slows a run. The loss chart and progress bar do not need it. It does not change the trained model.",
      beginner: "'Verbose' means chatty. Handy to see that training is alive and how long it will take.",
    },

    // ── Generate ──────────────────────────────────────────────────────────
    genModel: {
      title: "Model",
      what: "The trained .gguf model to generate text with, for example one you trained on the Train tab.",
      how: "The Train tab fills this in after a successful run. The model is loaded each time you press Generate, which is quick for small models.",
      beginner: "Generating means asking the model to keep writing from a starting text, one token at a time.",
    },
    genSeed: {
      title: "Seed text",
      what: "The beginning of the text the model should continue, for example 'Once upon a time'.",
      how: "The output starts with your seed. A character-level model (Vocab 0) needs a seed at least as long as its Context and only characters it saw in training. BPE models accept any text.",
      beginner: "The model never answers questions; it only predicts what comes next. A good seed sets the style and topic, just like the first sentence of a story.",
    },
    genChars: {
      title: "Chars",
      what: "How many new characters to generate after the seed (always characters, even for BPE models). Default 200.",
      how: "Time grows linearly with this number. The model may stop earlier if it emits its end-of-text token. Very long outputs from a tiny model tend to loop or ramble.",
      beginner: "200 characters is about two sentences. Small models stay coherent only for a short stretch.",
    },
    genTemp: {
      title: "Temperature",
      what: "How adventurous the word choice is. 0 means greedy: always the single most likely token, exactly as the command line does with --temp 0. Default 0.8.",
      how: "With 0 the text is fully repeatable and the gen seed has no effect. Low values (0.1 to 0.3) make the model pick the most likely token almost every time: safe and repetitive. Values around 0.7 to 1.0 give variety. High values (1.5 and up) give increasingly random text.",
      beginner: "Think of a dial between 'always say the obvious thing' and 'say something surprising'. For small models a middle setting is usually the best balance.",
    },
    genSeedNum: {
      title: "Gen seed (blank = random)",
      what: "A number that fixes the random choices during generation. Leave it empty for a different result each time.",
      how: "With the same model, seed text, temperature and gen seed, the output is exactly the same on every run and at any thread count. Use it to compare two settings fairly.",
      beginner: "Generation involves dice rolls. Fixing the seed fixes the dice, so you can repeat an experiment and get identical text.",
    },
    genMaxTokens: {
      title: "Max tokens (0 = use chars)",
      what: "A limit on the number of tokens to generate instead of characters. 0 keeps the Chars limit. Default 0.",
      how: "When set (or when a stop string is given) generation counts tokens, like --max-tokens on the command line; the seed text may be any length and the output never includes a stop string. Character-level models cannot use this mode.",
      beginner: "A token is a small piece of a word. 100 tokens is roughly 70 words of English.",
    },
    genTopK: {
      title: "Top-k (0 = off)",
      what: "Only the K most likely next tokens are considered. 0 disables the limit. Default 0.",
      how: "A value like 40 stops very unlikely tokens from ever being picked, even at a high temperature. 1 behaves like greedy decoding.",
      beginner: "The model ranks every possible next word. Top-k says 'only choose among the best K'.",
    },
    genTopP: {
      title: "Top-p (1.0 = off)",
      what: "Nucleus sampling: keep the smallest group of likely tokens whose probabilities add up to P and ignore the rest. From 0 up to 1; 1.0 is off. Default 1.0.",
      how: "0.9 is a common value. The group shrinks when the model is confident and grows when it is unsure.",
      beginner: "Instead of a fixed number of candidates, keep just enough candidates to cover 90 percent of the model's belief.",
    },
    genRep: {
      title: "Repetition penalty (1.0 = off)",
      what: "Makes tokens that were already used less likely. Must be above 0; 1.0 does nothing. Default 1.0.",
      how: "Values around 1.1 to 1.2 reduce loops such as 'the the the'. Large values make the text avoid ordinary repetition and become strange.",
      beginner: "A nudge against repeating yourself, a common habit of small language models.",
    },
    genStop: {
      title: "Stop strings (one per line)",
      what: "Text that ends generation as soon as the model writes it. The stop text itself is not part of the output. Empty lines are ignored.",
      how: "Type one stop string per line. Write \\n for a new line, \\t for a tab and \\\\ for a backslash, because a real new line would start the next stop string. Needs a BPE model (a vocabulary of 256 or more).",
      beginner: "Handy to cut the text at the end of a sentence or a paragraph instead of letting the model ramble on.",
    },
    genStream: {
      title: "Stream output live",
      what: "Shows the text piece by piece as it is produced instead of all at once at the end.",
      how: "The final text always matches the non-streaming result. Streaming only changes how it appears; it does not make generation faster.",
      beginner: "Like watching someone type rather than waiting for the finished letter.",
    },
    genVerbose: {
      title: "Verbose",
      what: "Sends extra diagnostic lines from the engine to the Engine log while generating.",
      how: "Mostly useful for debugging. It does not change the text that is generated.",
      beginner: "'Verbose' means the program explains what it is doing in more detail.",
    },

    // ── Evaluate ──────────────────────────────────────────────────────────
    evModel: {
      title: "Model",
      what: "The .gguf model to score. Each Evaluate press adds one row, so you can compare several models in one table.",
      how: "Works with models trained on the Train tab (the Train tab fills in the path after a run) and with imported Llama or Qwen .gguf files, which are scored like the eval-gguf command: the options in the box below apply to those only.",
      beginner: "Evaluating asks 'how surprised is the model by text it has never seen?'. Less surprise means a better model.",
    },
    evFile: {
      title: "Held-out text file",
      what: "A text file the model was not trained on, used to measure how well it generalises.",
      how: "If you also paste text in the box below, the pasted text is used and this file is ignored. The score is computed over every token with a sliding window; time is proportional to the text length.",
      beginner: "It is the exam paper. If the model saw these exact pages during training the score would be fake, so use text you set aside beforehand.",
    },
    evTestFile: {
      title: "Test text file (optional)",
      what: "A second text file, scored the same way and added to the table as its own 'test' row next to the held-out row.",
      how: "Use it when you split your data into three parts: training text, held-out text for choosing settings (the file above) and a final test file you only look at once. Leave it empty to score just one text.",
      beginner: "Think of the held-out file as practice exams and the test file as the final exam. Keeping them separate stops you from fooling yourself by tuning against the final one.",
    },
    evLossy: {
      title: "Decode invalid UTF-8 lossily",
      what: "When ticked, bytes in a text file that are not valid UTF-8 are replaced by a placeholder character instead of the file being refused.",
      how: "Off by default: a file with broken bytes is rejected with a clear message. The pasted-text box is always valid text. Replaced bytes change the text slightly, so the bits per byte differs a little from the original.",
      beginner: "Computers store text as bytes. If a file is damaged or in another encoding some bytes make no sense; lossy mode swaps those for a question-mark-like symbol so you can still get a score.",
    },
    evQuant: {
      title: "Precision (imported GGUF)",
      what: "How the weights of an imported Llama or Qwen model are held in memory while scoring: f32, int8 or int4. Ignored for Ferrum models. Default f32.",
      how: "f32 is exact and needs the most RAM. int8 and int4 use much less memory but round the weights, so the perplexity is slightly worse than the true value. Applying a fine-tune checkpoint always uses f32.",
      beginner: "Quantization stores each number with fewer bits, like saving a photo at lower quality: smaller and faster, slightly less exact. For a trustworthy score use f32 if it fits in your RAM.",
    },
    evCtx: {
      title: "Window tokens (0 = auto)",
      what: "How many tokens the imported model reads at once while scoring. 0 means the smaller of 512 and the model's own limit. Ignored for Ferrum models.",
      how: "Long texts are scored in overlapping windows so every token still gets history. A bigger window gives each token more context but takes more memory and time; it can never exceed the model's context length.",
      beginner: "The window is how much of the text the model can look at in one go, like the size of its short-term memory.",
    },
    evResume: {
      title: "Fine-tune checkpoint (imported GGUF)",
      what: "An optional .flck file from the Fine-tune tab. Its trained weights are applied to the imported model before scoring, so you can measure what fine-tuning changed. Ignored for Ferrum models.",
      how: "Score the same text once without and once with the checkpoint and compare the two rows. The model is loaded in f32 when a checkpoint is applied, which needs more RAM.",
      beginner: "A checkpoint is a save game of the fine-tuning. Loading it shows how well the tuned model does on text it has not seen.",
    },
    evForce: {
      title: "Load even if it may exceed available RAM",
      what: "Skips the safety check that refuses to load an imported model estimated to need more than 90 percent of your free RAM.",
      how: "Leave it off unless you know better: a model that does not fit makes the computer swap to disk and become very slow, or the load fails. A smaller precision is usually the better fix.",
      beginner: "The check protects you from freezing your computer. Ticking this box says 'I understand, try anyway'.",
    },
    evJsonPath: {
      title: "Save JSON to",
      what: "Where the Save JSON button writes the result of the latest evaluation.",
      how: "The file has the same keys as the command line's eval --json: perplexity, cross_entropy_nats, bits_per_token, bits_per_byte, tokens, bytes and seconds. If a test file was scored too, the file holds held_out and test objects with those keys.",
      beginner: "JSON is a plain-text format that other programs and spreadsheets can read, handy for keeping a record of your results.",
    },
    evText: {
      title: "Paste held-out text",
      what: "Text pasted here is scored instead of the file above (pasted text takes priority when both are given).",
      how: "Good for a quick test with a paragraph. For a meaningful number use at least a few thousand characters of unseen text, written in the same style as the training data.",
      beginner: "The result is perplexity: roughly 'how many words the model is torn between at each step'. 1 would be perfect; a model that learned nothing scores about its vocabulary size.",
    },

    // ── Models ────────────────────────────────────────────────────────────
    miPath: {
      title: "Model file",
      what: "The path of a Ferrum .gguf model to look inside.",
      how: "Inspect shows format, task, input and output sizes, tokenizer, number of BPE merges and layers. Nothing is changed. Use the GGUF tab for imported Llama or Qwen files.",
      beginner: "A quick way to check what a model file contains before using it, like looking at the label on a jar.",
    },

    // ── GGUF ──────────────────────────────────────────────────────────────
    ggPath: {
      title: "GGUF file",
      what: "The path of an external Llama or Qwen model in GGUF format that you want to inspect or run.",
      how: "Inspect reads only the file header, so it is quick even for huge files. Run then loads the weights (CPU only; a 1 billion parameter model decodes only a few tokens per second). Only the llama and qwen2 architectures can be run.",
      beginner: "People publish open models as .gguf files (for example on HuggingFace). A bigger model knows more but needs more memory and is slower on a CPU.",
    },
    ggPrompt: {
      title: "Prompt",
      what: "The text you give the model. For instruction-tuned models this is your question or request.",
      how: "With Chat format on, instruct models get their chat template applied automatically (single turn). Long prompts take time before the first new token appears, because every prompt token must be processed first.",
      beginner: "Unlike the small Ferrum models, these imported models can follow instructions, such as 'Write a haiku about rain'.",
    },
    ggQuant: {
      title: "Precision",
      what: "How the weights are stored in memory while the model runs: int8, int4 or f32.",
      how: "int4 needs the least RAM (about half of int8) and int8 is usually faster. f32 keeps the original precision but uses 4 times the memory of int8. Re-quantizing is lossy; f32 avoids that loss. Applying a fine-tune checkpoint forces f32.",
      beginner: "Quantization rounds each number to fewer bits, like saving a photo at lower quality: smaller and faster, slightly less accurate.",
    },
    ggMax: {
      title: "Max new tokens",
      what: "The most tokens the model will write. Default 64, at least 1.",
      how: "Generation stops earlier if the model emits its end-of-text token. Time grows linearly: each new token streams the whole model through memory once.",
      beginner: "A token is a piece of a word. 64 tokens is roughly 40 to 50 English words.",
    },
    ggTemp: {
      title: "Temperature",
      what: "How adventurous the word choice is. 0 means greedy: always the single most likely token. Default 0.8.",
      how: "With 0 the answer is the same every run. Lower is more predictable, higher is more varied. Combine with Top-k and Top-p to cut off unlikely tokens.",
      beginner: "A creativity dial: low for factual answers, higher for stories.",
    },
    ggTopK: {
      title: "Top-k (0 = off)",
      what: "Only the K most likely next tokens are considered. 0 disables the limit.",
      how: "A value like 40 prevents very unlikely tokens from ever being picked, even at a high temperature. 1 is the same as always taking the top choice.",
      beginner: "Imagine the model ranks every possible next word. Top-k says 'only choose among the best K'.",
    },
    ggTopP: {
      title: "Top-p (1.0 = off)",
      what: "Nucleus sampling: keep the smallest group of likely tokens whose probabilities add up to P, and ignore the rest. From above 0 up to 1; 1.0 is off.",
      how: "0.9 is a common value. The group size adapts: it is small when the model is confident and large when it is unsure, which is why many people prefer it to Top-k.",
      beginner: "Instead of a fixed number of candidates, it keeps just enough candidates to cover 90 percent of the model's belief.",
    },
    ggRep: {
      title: "Repetition penalty (1.0 = off)",
      what: "Makes tokens that were already used less likely. Must be above 0; 1.0 does nothing.",
      how: "Values around 1.1 to 1.2 reduce loops such as 'the the the'. Large values make the text avoid ordinary repetition and become strange.",
      beginner: "A nudge against saying the same thing again and again, a common habit of language models.",
    },
    ggSeed: {
      title: "Gen seed (blank = random)",
      what: "Fixes the random choices of the sampler. Empty means a different result each time.",
      how: "With the same file, settings, prompt and seed the output is reproducible. At temperature near zero the seed barely matters.",
      beginner: "Fixing the seed makes the dice rolls repeat, so you can rerun and compare.",
    },
    ggIds: {
      title: "Token IDs (only if the file has no tokenizer)",
      what: "A list of numbers separated by spaces, for example 1 234 56, used as the prompt instead of text.",
      how: "If this field is filled, it is used instead of the Prompt, even when the file has a tokenizer. The output then also comes back as numbers when there is no tokenizer to turn them into text.",
      beginner: "Models do not read letters; they read numbers (token IDs) from a lookup table. Normally the tokenizer does the conversion for you. Leave this empty unless you know you need it.",
    },
    ggResume: {
      title: "Fine-tune checkpoint (optional)",
      what: "A .flck file made on the Fine-tune tab. Its trained weights are overlaid on the base model before running.",
      how: "Forces f32 precision (a checkpoint holds 32-bit weights) so it needs about four times the RAM of int8. Leave empty to run the base model.",
      beginner: "Fine-tuning teaches an existing model your own text. This field lets you try the result without exporting a new file.",
    },
    ggChatFmt: {
      title: "Chat format (auto)",
      what: "Wraps your prompt in the model's chat template, the special tags instruct models were trained with, and stops at the end-of-turn token.",
      how: "The template is detected from the file's metadata or architecture. It handles a single turn only. Turn it off for base (non-instruct) models, which just continue raw text.",
      beginner: "Chat models expect messages marked 'user' and 'assistant'. This option adds those markers automatically so the model knows it is being asked a question.",
    },
    ggForce: {
      title: "Load even if it may exceed available RAM",
      what: "Skips the safety check that refuses to load a model estimated to use more than 90 percent of your free RAM.",
      how: "If the estimate is right, the computer will start swapping to disk (very slow) or the load fails. Only tick it if you know the estimate is too pessimistic.",
      beginner: "RAM is the computer's fast working memory. A model that does not fit in RAM makes the whole machine crawl.",
    },

    // ── Fine-tune ─────────────────────────────────────────────────────────
    ftPath: {
      title: "GGUF file",
      what: "The Llama or Qwen model (.gguf) that you want to adapt to your own text.",
      how: "The model is loaded at full 32-bit precision because training needs exact numbers. Training state takes about 4 times the weights, and saving the checkpoint peaks near 12 times the weights, so keep to small models (hundreds of millions of parameters at most on a laptop).",
      beginner: "Fine-tuning starts from a model that already knows a language and teaches it your style or topic. It is far cheaper than training from nothing, but still heavy on a CPU.",
    },
    ftCorpus: {
      title: "Training corpus (.txt)",
      what: "The plain text file the model is fine-tuned on.",
      how: "It is tokenized with the model's own tokenizer. Every epoch trains on every overlapping window of Sequence length tokens, so the work grows quickly with corpus size: keep the text small (tens of kilobytes) on a CPU.",
      beginner: "This text is what the model will start sounding like. A few hundred well-chosen lines in the style you want beat a huge messy file.",
    },
    ftOut: {
      title: "Output checkpoint (.flck)",
      what: "Where the fine-tuned weights and optimizer state are saved, as a .flck file.",
      how: "Use it on the GGUF tab (Fine-tune checkpoint field) to run the tuned model or on the Export tab to write a new GGUF. It must differ from the model and the corpus files.",
      beginner: "The original model is never modified. The checkpoint only stores the changes your fine-tuning made.",
    },
    ftEpochs: {
      title: "Epochs",
      what: "How many passes over the corpus. Default 3, at least 1.",
      how: "Time grows in direct proportion. Few epochs (1 to 5) are usually enough; many epochs on a small file make the model memorize it.",
      beginner: "One epoch is one full read of your text. Fine-tuning needs far fewer than training from scratch because the model already knows the language.",
    },
    ftLr: {
      title: "Learning rate",
      what: "The step size of the AdamW optimizer. Default 0.0001; must be above 0.",
      how: "Fine-tuning wants gentle steps, typically 0.00001 to 0.0003. Too high and the model forgets what it knew; too low and nothing changes.",
      beginner: "The model is already good. Big changes would damage it, so you adjust it with small careful nudges.",
    },
    ftBatch: {
      title: "Batch size",
      what: "How many windows go into each update. Default 8, at least 1.",
      how: "Bigger batches use more memory and give fewer updates per epoch. It also caps the number of useful threads.",
      beginner: "The model learns from several snippets at once and averages what it learned.",
    },
    ftSeq: {
      title: "Sequence length",
      what: "Tokens per training window. Default 64, at least 2; it is capped by the model's own maximum context length.",
      how: "Longer windows teach longer-range patterns but cost proportionally more time and memory. Each epoch trains on every overlapping window, so cost scales with this number twice (more windows to process and longer ones).",
      beginner: "How much text the model reads in one go while learning. Short windows are fine for style and vocabulary.",
    },
    ftWarmup: {
      title: "Warmup steps (0 = off)",
      what: "Number of steps over which the learning rate climbs from zero to its full value, followed by a cosine decay to zero. 0 keeps a constant rate.",
      how: "A schedule of a few dozen warmup steps makes fine-tuning smoother. The total step count is computed from epochs, corpus size and batch size.",
      beginner: "Start slowly, then speed up, then slow down again at the end: a gentle ramp makes training more stable.",
    },
    ftClip: {
      title: "Grad clip (0 = off)",
      what: "Limits the size of each update (the global gradient norm). Default 1.0; 0 turns it off.",
      how: "Protects the pretrained weights from a sudden large correction. Leave at 1.0 unless you have a reason.",
      beginner: "A safety limiter on how hard a single mistake can push the model around.",
    },
    ftWd: {
      title: "Weight decay",
      what: "A mild pull of all weights toward zero in every step (AdamW). Default 0.",
      how: "Small values such as 0.01 reduce overfitting to a small corpus. Leave at 0 for short fine-tunes.",
      beginner: "A gentle reminder not to rely too strongly on any single number the model has stored.",
    },
    ftDropout: {
      title: "Dropout",
      what: "The share of neurons randomly switched off while training, from 0 to 0.95. Default 0.",
      how: "Helps when a small corpus makes the model memorize it. Only active while training.",
      beginner: "Studying with part of your memory randomly muted makes what you learn more robust.",
    },
    ftThreads: {
      title: "Threads (0 = auto)",
      what: "Worker threads for training. 0 uses Ferrum's default (the number of CPU cores).",
      how: "Each extra thread holds its own copy of the gradients, which costs RAM about the size of the weights. Speed-up is sub-linear: one measurement on a small model gave 2.1 times with 3 threads and 3.4 times with 8.",
      beginner: "More threads use more CPU cores at once so the run finishes sooner, until memory or the number of cores gets in the way.",
    },
    ftSeed: {
      title: "Seed",
      what: "Starting value of the random generator that shuffles the training windows. Default 1337.",
      how: "The same seed and settings reproduce the same fine-tune. It is ignored when resuming, because the saved random state is restored instead.",
      beginner: "Fixing the seed makes the shuffle repeatable so experiments can be compared.",
    },
    ftResume: {
      title: "Resume checkpoint (optional)",
      what: "A .flck file from an earlier fine-tune to continue, including the optimizer's memory.",
      how: "The new run starts from those weights and counts steps cumulatively. Leave empty for a fresh fine-tune of the base model. Must be a different file from the output.",
      beginner: "Like loading a save game and training some more.",
    },
    ftQat: {
      title: "Quantization-aware (int8) fine-tuning",
      what: "Fine-tunes while simulating 8-bit rounding of the weights.",
      how: "Choose it if you will export the result as an int8 or similar quantized file, so the tuned model keeps its quality after rounding. It costs some speed.",
      beginner: "Prepares the model to survive being compressed afterwards.",
    },
    ftForce: {
      title: "Train even if the estimate exceeds available RAM",
      what: "Skips the guard that refuses to start when the estimated training memory is above 90 percent of free RAM.",
      how: "The guard counts 4 times the f32 weights, but saving the checkpoint can peak near 12 times, so a forced run on a large model usually ends up swapping or failing. Use only on purpose.",
      beginner: "If training does not fit into RAM, the computer slows to a crawl or the program is killed.",
    },

    // ── Export ────────────────────────────────────────────────────────────
    exPath: {
      title: "Source GGUF",
      what: "The Llama or Qwen .gguf file to convert.",
      how: "Inspect shows its shape and the RAM an f32 load needs. The whole model is loaded at f32 for the export, then re-encoded.",
      beginner: "Export writes a new copy of the model, in the format you choose. The original file is left alone.",
    },
    exQuant: {
      title: "Output type",
      what: "The number format of the weights in the new file: q8_0, q4_k, q5_k, q6_k, q4_0, q4_1, q8_1, f16 or f32.",
      how: "Lower bit counts give smaller files and faster loading at some quality loss: q8_0 is high quality, q4_k small and modern, q6_k near-lossless. f16 and f32 are lossless when the source is exact (fine-tuned f32 weights). Re-quantizing an already quantized file is lossy. Rows that do not fit a block size fall back to f16, and the summary tells you.",
      beginner: "It is like choosing JPEG quality for a photo. More bits keep more detail; fewer bits make a smaller file.",
    },
    exResume: {
      title: "Fine-tune checkpoint (optional)",
      what: "A .flck file from the Fine-tune tab to apply before exporting, so the new GGUF contains the tuned weights.",
      how: "Applying a checkpoint needs about 4 times the f32 model in RAM temporarily (weights, gradients and optimizer memory). Leave empty to export the base model unchanged.",
      beginner: "This is how you turn the result of your fine-tuning into a normal model file that other programs like Ollama or LM Studio can open.",
    },
    exOut: {
      title: "Output file (.gguf)",
      what: "Where the exported model is written.",
      how: "The write is atomic: a temporary file is renamed at the end, so a failed export leaves no broken file. After writing, the file is re-opened to prove it is valid and the tensor types are listed.",
      beginner: "Pick a new file name; do not overwrite your source model.",
    },
    exForce: {
      title: "Export even if the estimate exceeds available RAM",
      what: "Skips the guard that refuses to export when the estimated peak memory is above 90 percent of free RAM.",
      how: "The peak is the f32 model plus the output buffer, or 4 times the f32 model when a checkpoint is applied. Forcing a too-big export risks heavy swapping or a crash.",
      beginner: "A safety brake that you can override at your own risk.",
    },


    // ── Serve ─────────────────────────────────────────────────────────────
    svModel: {
      title: "Model to serve",
      what: "The Ferrum model file (a .gguf made on the Train tab) that the server loads and answers questions about.",
      how: "Type the path or press Browse. The whole file is read when you press Start server, and the model stays in memory until you press Stop server.",
      beginner: "A server is a program that waits for other programs to send it requests. Here the requests are 'continue this text' or 'how likely is this text'.",
    },
    svHost: {
      title: "Host",
      what: "Which network address the server listens on. 127.0.0.1 (also called localhost) means only programs on this computer can reach it. Default 127.0.0.1.",
      how: "Use 0.0.0.0 to let other computers on your network connect. Any host other than 127.0.0.1, ::1 or localhost needs a bearer token, and the server refuses to start without one.",
      beginner: "Think of it as which doors of the house are open. 127.0.0.1 opens only the door to your own room; 0.0.0.0 opens the front door to the whole network, so you then need a password (the token).",
    },
    svPort: {
      title: "Port",
      what: "The number that, together with the host, forms the address programs connect to, for example 127.0.0.1:8080. Default 8080; 0 means pick any free port.",
      how: "If another program already uses the port, Start server fails and says so; choose another number or 0. The address that was chosen is shown after the start.",
      beginner: "If the host is a building, the port is the door number. Use 0 when you do not care which door; the Studio tells you which one it got.",
    },
    svMaxBatch: {
      title: "Max batch",
      what: "The most texts one request may contain. A request with more is refused with an error before any work is done. Default 64.",
      how: "Raise it for a harness that sends big batches, lower it to keep requests small. Each text is still processed one after the other.",
      beginner: "A limit that protects the server from one huge request taking all your computer's time.",
    },
    svMaxChars: {
      title: "Max characters per text",
      what: "The longest single text (prompt, context or continuation) the server accepts, counted in characters. Default 65536.",
      how: "Longer texts are refused with an error. Models only look at their context window anyway, so a very high value rarely helps.",
      beginner: "Another safety limit: a text longer than this is turned away instead of being worked on for ages.",
    },
    svMaxBody: {
      title: "Max request body (bytes)",
      what: "The biggest request the server will even read, in bytes. Default 8388608, which is 8 MiB.",
      how: "A bigger request is refused with error 413 before it is read. Keep it above max batch times max characters if you use both near their limits.",
      beginner: "A byte is one character of plain English text, so 8 MiB is roughly an 8 million letter message.",
    },
    svMaxTokens: {
      title: "Max tokens per generation",
      what: "The most new tokens a single generate request may ask for. Default 4096.",
      how: "A request that asks for more is refused. Generating is slow on a CPU, so a lower value keeps each answer quick.",
      beginner: "A token is a small piece of a word. 4096 tokens is several pages of text; small models rarely stay sensible that long.",
    },
    svToken: {
      title: "Bearer token",
      what: "A secret word that clients must send in an Authorization header (Bearer, then the token) with every request except GET /health. Leave it empty to turn the check off, which is only allowed on this machine.",
      how: "Type your own or press Generate. It is kept in memory while the server runs and is never saved by the Studio, not even in the remembered settings, so it is empty again after a restart. There is no HTTPS, so on a shared network the token keeps honest users out but does not hide traffic from someone who can watch it.",
      beginner: "A token is a password for programs. Clients must send it with each request, like showing a membership card at the door.",
    },
    svShowToken: {
      title: "Show token",
      what: "Shows the token in the field instead of dots, so you can read or copy it.",
      how: "Tick it right after pressing Generate to copy the new token before you start the server. Untick it when others can see your screen.",
      beginner: "Normally a password is hidden as dots while you type. This switch lets you look at it.",
    },
    // ── System ────────────────────────────────────────────────────────────
    sysOn: {
      title: "Monitoring on",
      what: "Turns the live CPU and memory readout on or off (also the small gauges in the top bar).",
      how: "While on, the app asks the system for the numbers every Interval seconds. The cost is negligible, but you can turn it off while a long run needs every cycle.",
      beginner: "It shows how busy your computer is, which tells you whether training is really using all your CPU cores.",
    },
    sysInterval: {
      title: "Interval (s)",
      what: "Seconds between refreshes of the monitor. Default 1.5, minimum 0.5.",
      how: "Changing it restarts the timer. Shorter intervals look smoother but use slightly more CPU, and anything below 0.5 is raised to 0.5. The value is in seconds and may have a decimal part.",
      beginner: "How often the numbers update. One to two seconds is plenty for watching a long training run; you do not need it faster.",
    },

    // ── Buttons ───────────────────────────────────────────────────────────
    dsCatalogLoad: {
      title: "Refresh list",
      does: "Reloads the built-in list of small, CPU-friendly datasets (HuggingFace and Kaggle) shown below.",
      how: "The list is built into the app, not fetched from the internet, and keeps only entries of at most 200 MB. It is also loaded once when the window opens; a failure is shown under the catalog.",
      kind: "instant",
    },
    dsCatalogDownload: {
      title: "Download (catalog item)",
      does: "Downloads this dataset file into the download folder, with a size cap of 512 MiB, and then fills in the local file field.",
      kind: "download",
      beginner: "How long it takes depends on your internet connection and the file size shown on the card.",
    },
    dsHfDownload: {
      title: "Fetch HF",
      does: "Downloads the named file from the named HuggingFace dataset into the download folder (up to 512 MiB, through a temporary .part file).",
      kind: "download",
      beginner: "Check the repo, file and folder fields first; a wrong name gives a clear error message.",
    },
    dsKgDownload: {
      title: "Fetch Kaggle",
      does: "Downloads the named file from the named Kaggle dataset into the download folder. Needs Kaggle credentials.",
      how: "It needs a download folder, the owner/slug and the file name. Credentials come from KAGGLE_USERNAME and KAGGLE_KEY or from ~/.kaggle/kaggle.json, and the saved file is put into the local file field.",
      kind: "download",
    },
    dsDownload: {
      title: "Download",
      does: "Downloads the text at the Source URL, up to the Max MB limit, and puts it in the Raw text box.",
      kind: "download",
      beginner: "The time depends on your internet connection and the size of the file.",
    },
    dsLoadFile: {
      title: "Load file",
      does: "Reads the local file (up to 16 MiB) into the Raw text box.",
      how: "The path comes from the local file field; invalid UTF-8 bytes are replaced instead of refused. Loading replaces whatever is in the Raw text box, and a notice reports how many characters arrived.",
      kind: "load_file",
    },
    dsClean: {
      title: "Clean and preview",
      does: "Applies the cleaning options to the Raw text, shows the first 4,000 characters and prints statistics: characters, words, lines, bytes and the number of distinct characters.",
      kind: "clean_text",
      beginner: "Nothing is saved yet. The distinct-character count is the vocabulary size a character-level model would need.",
    },
    dsSave: {
      title: "Save corpus",
      does: "Writes the cleaned text to the Save to path (creating folders as needed) and fills the Train tab's Corpus field.",
      how: "It saves the whole cleaned text, not only the 4000-character preview, and fails with a message when nothing was cleaned yet. The Train Corpus field is filled only when it is empty.",
      kind: "instant",
    },
    trStart: {
      title: "Train",
      does: "Trains a new transformer language model on the corpus with the settings above and saves it as a .gguf file. The loss chart and progress bar update live. The pre-flight check above shows the parameter count, the memory needed and whether llama.cpp can run the result; Start is disabled when the run would not fit in RAM.",
      kind: "train",
      beginner: "Training is the slow part: the model reads the text many times, adjusting its numbers to predict the next token better. The time estimate comes from your CPU cores, the model size and the corpus size; it can be off by a factor of two either way.",
    },
    genStart: {
      title: "Generate",
      does: "Loads the model and writes new text that continues the seed, showing it live when streaming is on.",
      how: "Chars or Max tokens set the length; Temperature, Top-k, Top-p and Repetition penalty shape the choice of each token; Stop strings end the text early. The button is disabled while a generation runs.",
      kind: "generate",
    },
    evRun: {
      title: "Evaluate and add row",
      does: "Scores the model on the held-out text (and the test file, if given) and adds a row for each with perplexity, cross-entropy, bits per token, bits per byte, the time taken and a verdict to the table.",
      kind: "evaluate",
      beginner: "Perplexity is the key number. Lower is better, and the uniform baseline column is what a model that learned nothing would score.",
    },
    trCancel: {
      title: "Stop training",
      does: "Ends the running training after the current step. The model trained so far is saved to the Model output path (and the checkpoint is written if you set one), so you keep what was learned and can resume.",
      kind: "instant",
      beginner: "Handy when the loss has stopped improving or you started with too many epochs. A stopped model is simply less trained, not broken.",
    },
    genCancel: {
      title: "Stop",
      does: "Ends the running generation after the current piece of text. What was produced so far stays in the output box.",
      how: "The button appears only while a generation runs. The status line then reads stopped, with the number of characters returned.",
      kind: "instant",
    },
    ggCancel: {
      title: "Stop",
      does: "Ends the running GGUF generation before the next token. The text produced so far is shown. While the model is still loading, it stops once loading finishes.",
      kind: "instant",
      beginner: "On a CPU a large model writes only a few tokens per second, so stopping early saves a lot of waiting.",
    },
    ftCancel: {
      title: "Stop fine-tuning",
      does: "Ends the running fine-tune after the current step and still writes the checkpoint, so you can continue later with the Resume checkpoint field or try the partly tuned model on the GGUF tab.",
      kind: "instant",
    },
    exCancel: {
      title: "Cancel export",
      does: "Cancels the export between its phases (opening, loading, applying the checkpoint). Nothing is written when it is cancelled. Once the final encode-and-write phase has started it cannot be interrupted and the file is finished.",
      kind: "instant",
    },
    evSaveJson: {
      title: "Save JSON",
      does: "Writes the latest evaluation (every number of the last run, including the test row if there was one) to the Save JSON path as a JSON file with the same keys as the command line's eval --json.",
      kind: "instant",
      beginner: "Run an evaluation first; this only saves what is already computed, nothing is scored again.",
    },
    evClear: {
      title: "Clear table",
      does: "Removes all rows from the evaluation table. No file is touched.",
      how: "It also forgets the stored copy of the table that would otherwise come back after a restart. After clearing, Save JSON has nothing to save until the next evaluation.",
      kind: "instant",
    },
    miInspect: {
      title: "Inspect / reload",
      does: "Reads the model file and shows its format, task, sizes, tokenizer and number of layers.",
      how: "It reads the file named in Model file and shows the result as cards below; a file that is not a valid Ferrum model shows an error instead. The file itself is never changed.",
      kind: "model_info",
    },
    ggInspect: {
      title: "Inspect",
      does: "Reads only the GGUF header and shows the architecture, shape, tokenizer and the RAM needed at int4, int8 and f32. Warns when the model exceeds this machine's estimated limits.",
      how: "The warning appears only after the Capable tab has measured this machine; until then a notice suggests running that check.",
      kind: "gguf_info",
    },
    ggRun: {
      title: "Run",
      does: "Loads the GGUF model at the chosen precision and generates text from the prompt. If a Capable check was done and the model looks too big, it asks first.",
      kind: "gguf_run",
      beginner: "On a CPU this is slow: the model must read all its weights for every token. The estimate adds loading, reading the prompt and writing the new tokens.",
    },
    ftRun: {
      title: "Fine-tune",
      does: "Adapts the GGUF model to your corpus with the AdamW optimizer and saves a .flck checkpoint, with live loss updates.",
      kind: "gguf_finetune",
      beginner: "Fine-tuning a real model on a CPU can take hours or days even for small corpora; the estimate shows what to expect and warns when RAM is short.",
    },
    exInspect: {
      title: "Inspect",
      does: "Reads only the GGUF header and shows its architecture, shape and the memory an export would need.",
      how: "The cards show architecture, GGUF version and tensor count, shape, tokenizer, the f32 memory an export loads and the free RAM. After a Capable check, an oversized model also opens a warning.",
      kind: "gguf_info",
    },
    exExport: {
      title: "Export",
      does: "Converts the model (with the fine-tune checkpoint applied, if given) to the chosen output type and writes a new GGUF file, with progress for each phase.",
      how: "It is disabled while an export runs; Cancel export stops it and nothing is written. After a Capable check, a model over the limits opens a warning that asks first.",
      kind: "gguf_export",
    },
    capCheck: {
      title: "Check this machine",
      does: "Runs a short benchmark of memory speed and matrix-multiplication speed and estimates which model sizes this machine can load, train, fine-tune and run. In the release app the result also makes every time estimate in this window more accurate.",
      kind: "capability",
      beginner: "The numbers are rough upper bounds, not promises. Close other heavy programs for a fair reading.",
    },
    resetSettings: {
      title: "Reset settings",
      does: "Forgets everything the Studio remembered between runs (the values in the forms, the folders the file pickers start in and the Evaluate table) and puts every field back to its starting value. No file on disk is touched.",
      kind: "instant",
      beginner: "The Studio saves what you type in this browser profile so the next start looks the way you left it. This button wipes that memory and starts fresh.",
    },
    termClear: {
      title: "Clear",
      does: "Empties the Engine log panel. It does not affect any running job.",
      how: "New lines keep arriving after it: engine logs and messages from the Studio. The panel keeps only the latest 2000 lines on its own anyway.",
      kind: "instant",
    },
    ggWarnCancel: {
      title: "Cancel",
      does: "Closes the warning and does not run the oversized model.",
      how: "Escape does the same. After Inspect, cancelling only closes the dialog; before Run or Export it keeps the job from starting, and the status line reads cancelled.",
      kind: "instant",
    },
    ggWarnProceed: {
      title: "Proceed anyway",
      does: "Continues despite the warning that the model exceeds this machine's estimated limits. The job may be very slow or fail.",
      how: "After Inspect it only closes the dialog. Before Run or Export, the job starts as if no limit were crossed. The dialog lists every limit the model crosses.",
      kind: "instant",
    },
    capDialogClose: {
      title: "Close",
      does: "Closes the machine capability report. The numbers stay on the Capable tab.",
      how: "Escape closes it too. The bounds it measured stay in use: the GGUF and Export tabs check model sizes against them.",
      kind: "instant",
    },

    // Tabs
    // ── Serve: buttons ────────────────────────────────────────────────────
    svGenToken: {
      title: "Generate token",
      does: "Fills the token field with a new random token of 48 letters and digits made by your system's secure random generator, and shows it so you can copy it. Nothing is saved or sent anywhere.",
      kind: "instant",
      beginner: "Handy when you want a strong password but do not want to invent one. Copy it somewhere safe before you start the server.",
    },
    svStart: {
      title: "Start server",
      does: "Loads the model, opens the port and starts answering HTTP requests, then shows the address, a request counter and a log of every request. It refuses to start when a server is already running, a limit is below 1, the port is taken, or the host is not this machine and no token is set.",
      kind: "serve",
      beginner: "Loading reads the whole file and takes a moment for a big model. While the server runs the model stays in memory.",
    },
    svStop: {
      title: "Stop server",
      does: "Stops the server after the requests that are running have finished, closes the port and frees the model's memory. Waits at most 10 seconds for a stuck request before cutting it off.",
      kind: "instant",
    },
    svCopyUrl: {
      title: "Copy URL",
      does: "Copies the address of the running server, for example http://127.0.0.1:8080, to the clipboard.",
      how: "The button is visible only while the server runs. If the clipboard is blocked, a notice asks you to select the address and copy it by hand.",
      kind: "instant",
    },
    svCopyCurl: {
      title: "Copy curl example",
      does: "Copies a ready-made curl command that asks the running server to generate a few tokens. When a token is set the command reads it from the FERRUM_TOKEN environment variable, so your token is not in the copied text.",
      kind: "instant",
      beginner: "curl is a command-line tool that sends a request. Paste the command into a terminal to see the server answer.",
    },

    // ── Export: run in Ollama / LM Studio ─────────────────────────────────
    rxPath: {
      title: "Ferrum model for Ollama or LM Studio",
      what: "The .gguf file of a model trained with Ferrum using a byte-level BPE vocabulary (vocab 256 or more), which is stored as a gpt2 GGUF that other programs can run.",
      how: "It is filled in after a training run that used BPE. Character-level models (vocab 0) and imported llama models are refused before any program starts, with a message that says what to do; an imported llama model needs a Modelfile written by hand (Manual section 9.5). The file is only read, never changed.",
      beginner: "Ollama and LM Studio are free programs for chatting with models on your own computer. They can run a Ferrum model too, as long as it was trained the standard way (BPE tokenizer).",
    },
    rxPathBrowse: browse("Ferrum model"),
    rxName: {
      title: "Ollama model name",
      what: "The name the model gets inside Ollama, used for ollama create and later for ollama run. Letters, digits and . _ : / - are allowed; it may not start with a dash.",
      how: "Names like ferrum-tiny or me/ferrum:v1 work. Creating a model with a name that already exists replaces it in Ollama. The name is not used for LM Studio.",
      beginner: "It is just a label you choose. Afterwards you start the model in a terminal with: ollama run THE-NAME.",
    },
    rxRepo: {
      title: "LM Studio repo",
      what: "The publisher/name under which LM Studio lists the model, for example ferrum/my-model. Empty means ferrum followed by the file name without .gguf.",
      how: "Exactly one slash, with text on both sides. The file is linked into LM Studio, not copied, so do not move or delete it afterwards. Importing the same file twice is not an error.",
      beginner: "LM Studio organises models like a library: a publisher folder and a model folder. Leave this empty and it is filed under ferrum, which is fine.",
    },
    rxOllamaBin: {
      title: "ollama program",
      what: "Optional path to the ollama program. Empty means: the FERRUM_OLLAMA environment variable, then your PATH, then ~/.local/bin/ollama.",
      how: "Fill it in only if Ollama is installed somewhere unusual. The path must exist. If no program is found the message lists all four places that were searched.",
      beginner: "Your computer finds programs by name in a list of folders. If Ollama is installed in an odd place that is not on the list, point to it here.",
    },
    rxOllamaBinBrowse: browse("ollama program"),
    rxLmsBin: {
      title: "lms program",
      what: "Optional path to the lms program that comes with LM Studio. Empty means: the FERRUM_LMS environment variable, then your PATH, then ~/.lmstudio/bin/lms.",
      how: "Fill it in only if lms is installed somewhere unusual. The path must exist. LM Studio itself does not need to be open, but it must have been run once so lms exists.",
      beginner: "lms is the command-line helper installed by LM Studio. This button uses it to add the model to LM Studio's library.",
    },
    rxLmsBinBrowse: browse("lms program"),
    rxModelfileOut: {
      title: "Modelfile to write",
      what: "Where Write Modelfile only saves the Modelfile text. Needed only for that button.",
      how: "The file is written exactly as export-ollama would hand it to ollama create, so you can read it, edit it, or run ollama create yourself later. Nothing is started and Ollama is not contacted.",
      beginner: "A Modelfile is a short recipe that tells Ollama where the model file is and how to run it. You can keep the recipe and use it yourself.",
    },
    rxModelfileOutBrowse: browse("Modelfile", "save"),
    rxModelfile: {
      title: "Modelfile",
      what: "The exact Modelfile text for Ollama: the absolute model path, the prompt template, the stop word and the context length read from the model.",
      how: "Filled in by Show Modelfile and commands and by every Ollama export. Read-only here; use Write Modelfile only to save it and edit the copy.",
      beginner: "This is the recipe Ollama receives. The FROM line names your model file; the other lines say how to talk to it.",
    },
    rxPreview: {
      title: "Show Modelfile and commands",
      does: "Checks the model, then shows the exact Modelfile and the commands that Export to Ollama and Import into LM Studio would run, and which program was found. Nothing is started and neither program is contacted.",
      kind: "model_info",
      beginner: "A safe dry run. Use it to see what will happen before you press the real buttons.",
    },
    rxOllama: {
      title: "Export to Ollama",
      does: "Writes a temporary Modelfile and runs ollama create with your name, then shows the Modelfile, the command and Ollama's own output. The Ollama server must be running (ollama serve); if it is not, the message says so.",
      kind: "runtime_export",
      how: "Same as the command line: train_transformer export-ollama. A running ollama create cannot be stopped from here.",
      beginner: "After it succeeds, open a terminal and type: ollama run followed by your model name.",
    },
    rxLms: {
      title: "Import into LM Studio",
      does: "Runs lms import to link the model into LM Studio, then asks lms ls for its model key and shows both commands and the key. If LM Studio is slow to list it, this waits up to about 15 seconds.",
      kind: "runtime_export",
      how: "Same as the command line: train_transformer export-lmstudio. The file is linked, not copied.",
      beginner: "Afterwards the model shows up in LM Studio's list of models, or you can load it with lms load and the shown key.",
    },
    rxWriteModelfile: {
      title: "Write Modelfile only",
      does: "Saves the Modelfile to the file you chose and stops: no program is started and Ollama is not contacted. The same text is shown in the Modelfile box.",
      kind: "runtime_export",
      how: "Same as export-ollama with the modelfile-only option.",
      beginner: "Useful when Ollama is on another computer: copy the model and the recipe there and run ollama create yourself.",
    },
    rxCancel: {
      title: "Stop export",
      does: "Stops the export before the next program starts, or while waiting for LM Studio's listing. A program that is already running (ollama create) cannot be interrupted and finishes.",
      how: "It appears only while an export runs; the status line then reads stopped.",
      kind: "instant",
    },
    // ── Train: tokenize a corpus ──────────────────────────────────────────
    tkInput: {
      title: "Text file or folder to tokenize",
      what: "The text to turn into a token file: one file, or a folder (every file inside it, in sorted order, hidden files skipped).",
      how: "Documents inside a file are separated by the separator below. The text is read in small pieces and the ids are written as they are produced, so the corpus may be larger than your memory. Invalid UTF-8 stops the run with the file and byte position; the Prepare tab can repair it.",
      beginner: "Tokenizing means converting text into the numbers the model actually reads. Doing it once and saving the result lets you train repeatedly on a huge text without ever loading the text again.",
    },
    tkInputBrowse: browse("text file"),
    tkInputDirBrowse: browse("text folder", "dir"),
    tkOut: {
      title: "Token file to write",
      what: "Where the token file is saved. It is a GGUF file of numbers, about 2 bytes per token, usually well under the size of the text.",
      how: "It is written to a temporary file and renamed when complete, so a stopped or failed run leaves nothing behind. Afterwards the Corpus field above is set to it. Use only with the tokenizer named in the Tokenizer file field.",
      beginner: "This is the numbers version of your text. Keep it together with the tokenizer file it was made with; the pair is what training needs.",
    },
    tkOutBrowse: browse("token file", "save"),
    tkSep: {
      title: "Document separator (tokenize)",
      what: "The text that separates one document from the next inside a file, so an end-of-text marker can be placed between them. It starts as the standard end-of-text marker (see the default text of the field).",
      how: "Use the same separator as when you prepared the data. The marker between documents is added by the tokenizer, including between files, exactly as the command line does.",
      beginner: "The end-of-text marker tells the model where one story ends and the next begins, so it does not learn to blend them.",
    },
    tkRun: {
      title: "Tokenize corpus",
      does: "Reads the text from disk, converts it to token numbers with the tokenizer named in the Tokenizer file field, and writes the token file with progress. Then the Corpus field is set to it so Train can use it.",
      kind: "tokenize",
      how: "The same as the train_transformer tokenize command, so the numbers are the ones the command line gives. Training from the token file gives the same model as training from the text.",
      beginner: "Run it once per corpus and tokenizer. It needs a tokenizer file that already exists, which a first training run with the Tokenizer file field filled in creates.",
    },
    tkCancel: {
      title: "Stop tokenizing",
      does: "Stops the tokenizing between documents. No token file is written and the text is untouched.",
      how: "It appears only while tokenizing runs; the status line then reads stopped, and the Train Corpus field keeps its value.",
      kind: "instant",
    },
    // ── Prepare ───────────────────────────────────────────────────────────
    dpInputs: {
      title: "Input files and folders",
      what: "The text files to prepare, one path per line. A folder means every file inside it, including sub-folders (hidden files and the output folder are skipped).",
      how: "Type or paste paths, or use Add file and Add folder. Every file is read as it is, whatever its extension, so a stray picture or archive in a folder counts as invalid text. Files are read in sorted order in small pieces straight from disk, so a corpus larger than your memory is fine.",
      beginner: "This is your raw material. If you have one big book as a text file, add just that file. If you have a folder of many small texts, add the folder and every text in it becomes part of the dataset.",
    },
    dpAddFile: browse("input file", "add"),
    dpAddDir: browse("input folder", "adddir"),
    dpOut: {
      title: "Output folder",
      what: "The folder where train.txt, val.txt, test.txt and manifest.json are written. It is created if it does not exist.",
      how: "Files are first written next to their final names with a .part ending and renamed only when the whole run succeeded, so a failed or stopped run leaves any earlier files in this folder exactly as they were. The folder must not be one of the inputs' files, but it may be inside an input folder (it is skipped).",
      beginner: "Pick an empty or new folder, for example prepared inside your project. You will point the Train tab at train.txt in this folder afterwards.",
    },
    dpOutBrowse: browse("output folder", "dir"),
    dpVal: {
      title: "Validation share",
      what: "The fraction of documents kept aside as the validation set, between 0 and just under 1. Default 0.05, which is 5 percent.",
      how: "Each document goes to train, validation or test by a fingerprint of its own text and the split seed, so the same document always lands in the same set and adding documents later does not move the old ones. With few documents a set can end up empty. Validation and test shares together must stay below 1.",
      beginner: "Like keeping a few exam questions hidden from a student. Training never sees the validation text, so the Evaluate tab can tell you how the model does on text it has not memorised.",
    },
    dpTest: {
      title: "Test share",
      what: "The fraction of documents kept aside as the final test set. Default 0.05, which is 5 percent.",
      how: "Chosen the same way as the validation share. Use the validation set while you tune a model and the test set once at the end, so your final score is not tuned to it. Set it to 0 if you do not want a test file (it is then empty).",
      beginner: "The test set is the final exam. You only look at it when you are done experimenting, so the score is an honest one.",
    },
    dpSeed: {
      title: "Split seed",
      what: "A number mixed into the fingerprint that decides which set each document goes to. Default 42.",
      how: "The same seed and the same documents always give the same split. Change it to get a different random-looking split of the same data. It is written in the manifest, so a run can be repeated exactly.",
      beginner: "Think of it as the shuffle number. Keep it the same when you want to compare experiments fairly, change it when you want a different shuffle.",
    },
    dpMin: {
      title: "Minimum characters",
      what: "Documents shorter than this many characters (after cleaning) are dropped. Default 1, which only removes empty ones.",
      how: "Counts characters, not bytes. Dropped documents are counted as too short in the result. Raising it removes fragments such as headings and page numbers that teach a model little.",
      beginner: "A document with five letters cannot teach a model how sentences work. A minimum of 50 or so throws away scraps and keeps real text.",
    },
    dpMax: {
      title: "Maximum characters",
      what: "Documents longer than this many characters (after cleaning) are dropped. 0, the default, keeps documents of any length.",
      how: "Useful when a few giant documents (a whole book in one file) would dominate a small dataset. Dropped documents are counted as too long. Nothing is cut: a document is kept whole or dropped whole.",
      beginner: "If one file is a thousand times longer than the others, the model mostly learns that one file. A maximum keeps the mix balanced. Leave it at 0 if unsure.",
    },
    dpSep: {
      title: "Document separator",
      what: "The text that separates one document from the next inside a file. It starts as the standard end-of-text marker (see the default text of the field).",
      how: "A file with no separator in it is one single document, which is why a single book needs a separator or more files to be split into train, validation and test. Clearing the field makes every file exactly one document. The splits you get back are joined with the standard marker.",
      beginner: "A split by whole documents keeps stories or articles intact. If your file is one long text, you can use a blank line or a line of dashes between articles as the separator, if you type that here.",
    },
    dpDedup: {
      title: "Drop duplicate documents",
      what: "Keeps only the first copy of each document that appears more than once. Off by default.",
      how: "Compares a short fingerprint of each cleaned document, so only the fingerprints are held in memory. Dropped copies are counted as duplicates. Removing repeats stops the same text from landing in both train and test, which would flatter the score.",
      beginner: "If the same story appears twice, the model would see it twice and the test could contain something it already memorised. Dropping copies keeps the evaluation honest.",
    },
    dpNfkc: {
      title: "Unicode NFKC",
      what: "Rewrites look-alike characters into their plain forms, such as the single ligature character for fi into the two letters f and i. Off by default.",
      how: "Applied after invisible characters are removed and before punctuation is folded. It shrinks the number of distinct characters, which helps small vocabularies. It can change text that relies on special forms, such as superscripts.",
      beginner: "The same word can be typed with different hidden codes that look identical on screen. NFKC makes them the same, so the model does not treat one word as several.",
    },
    dpLower: {
      title: "Lowercase",
      what: "Converts every document to lowercase. Off by default.",
      how: "Shrinks the vocabulary, which helps tiny models, but generated text will be all lowercase. Applied together with the other cleaning steps, before the length filters.",
      beginner: "Capital letters double the number of letters a model must learn. For a first small model, lowercase text is easier to learn and still readable.",
    },
    dpGutenberg: {
      title: "Strip Project Gutenberg boilerplate",
      what: "Removes the license header and footer that Project Gutenberg puts around each book, and makes each book in a file its own document. Off by default.",
      how: "Looks for the START and END markers of Gutenberg texts. A file may hold several books; each becomes one document. Text without the markers passes through unchanged.",
      beginner: "Gutenberg books start with pages of legal text. Left in, a small model would learn to write legal text, so it is better removed.",
    },
    dpLossy: {
      title: "Replace invalid UTF-8 bytes",
      what: "Instead of stopping at the first byte that is not valid text, replaces such bytes with the replacement character and carries on. Off by default.",
      how: "Without it the run stops with an error naming the file and the byte position, so you can fix the file. With it, documents that needed repairs are still kept and the number of repaired documents is shown in the result.",
      beginner: "Some old files use a different character encoding and contain bytes that are not valid text. Strict mode tells you about them; this switch papers over them, at the price of a few odd characters.",
    },
    dpRun: {
      title: "Prepare",
      does: "Reads the inputs from disk, cleans and filters every document, splits them into train, validation and test files by a fingerprint of the text, and writes the files plus manifest.json with checksums. Shows progress and a summary of what was kept and rejected.",
      kind: "prepare",
      how: "It does exactly what the train_transformer prepare command does, so you can repeat a run on the command line from the arguments recorded in the manifest.",
      beginner: "Nothing in your inputs is changed. New files are written to the output folder, and only when everything worked.",
    },
    dpCancel: {
      title: "Stop preparing",
      does: "Stops the preparation between documents. No output file is changed: files from an earlier run in the same folder stay as they were and no partial files are left behind.",
      kind: "instant",
    },
    dpUse: {
      title: "Use the splits in Train and Evaluate",
      does: "Fills the Train tab's corpus with train.txt and the Evaluate tab's held-out and test files with val.txt and test.txt from the run that just finished (empty splits are skipped).",
      how: "It appears after a run that put at least one document into train.txt.",
      kind: "instant",
    },
    "tab-datasets": tab("Datasets", "download, load, clean and save the text your model learns from.",
      "It holds the dataset catalog, the HuggingFace and Kaggle fetchers, the URL download, the local file loader, the cleaning options and the save box. Nothing here trains a model; it produces a plain text corpus file."),
    "tab-prepare": tab("Prepare", "clean, deduplicate and split text files into train, validation and test sets with a manifest.",
      "It runs the same code as the train_transformer prepare command on files and folders read straight from disk, and writes train.txt, val.txt, test.txt and manifest.json into the output folder."),
    "tab-train": tab("Train", "train a new small transformer language model from scratch on your corpus.",
      "It holds the model shape, the optimizer settings, the optional tokenize step and a pre-flight check of parameters and peak RAM. The model is saved as a GGUF file when training ends or is stopped."),
    "tab-generate": tab("Generate", "write new text with a trained model, optionally streamed live.",
      "Point it at a Ferrum model file, type a seed text and choose the sampling settings; Stop ends a run early and keeps the text produced so far. Imported Llama or Qwen models run on the GGUF tab."),
    "tab-evaluate": tab("Evaluate", "score models on text they have not seen, using perplexity.",
      "Each run adds a row with cross-entropy, bits per token and per byte, perplexity and the uniform baseline, for a held-out text and an optional test file. The table is remembered across restarts and can be saved as JSON."),
    "tab-models": tab("Models", "inspect a model file.",
      "Point it at a Ferrum model file and press Inspect to see its size, format, name, task, input and output dimensions, tokenizer, BPE merge count and number of layers. Nothing is changed on disk."),
    "tab-gguf": tab("GGUF", "inspect and run an external Llama or Qwen model.",
      "It reads the GGUF header for architecture, shape, tokenizer and RAM needs, then generates from a prompt at int8, int4 or f32 precision, optionally with a fine-tune checkpoint laid over the weights. Everything runs on the CPU."),
    "tab-finetune": tab("Fine-tune", "adapt an external GGUF model to your own text.",
      "AdamW trains the imported model on a plain text corpus and writes a .flck checkpoint, with a loss chart and a Stop button. The GGUF, Evaluate and Export tabs can then load that checkpoint on top of the original file."),
    "tab-export": tab("Export", "write a (fine-tuned) GGUF model to a new file in another number format.",
      "Choose one of nine output types from q8_0 to f32 and an optional fine-tune checkpoint; the new file runs in llama.cpp, Ollama and LM Studio. The block at the bottom hands a Ferrum model to Ollama or LM Studio."),
    "tab-serve": tab("Serve", "run an HTTP server for one model so other programs can use it.",
      "It starts the same server as the ferrum-serve command inside this window, with limits per request, an optional bearer token, a request counter, a copyable curl example and a log of every request."),
    "tab-system": tab("System", "live CPU, memory and thread load.",
      "It polls the backend every 1.5 seconds by default, at an interval you can change, and shows total CPU load, memory in use, the Ferrum worker thread count and one bar per CPU core. Monitoring can be switched off."),
    "tab-capable": tab("Capable", "benchmark this machine and see which model sizes it can handle.",
      "Check this machine runs a micro-benchmark of about one to two seconds and estimates the model sizes this machine can load, train, fine-tune and run. The GGUF and Export tabs use the result to warn about oversized models."),

    // Browse buttons
    dsDestDirBrowse: browse("download folder", "dir"),
    dsFileBrowse: browse("local text file"),
    dsOutBrowse: browse("output corpus file", "save"),
    trCorpusBrowse: browse("training corpus"),
    trModelBrowse: browse("model output file", "save"),
    genModelBrowse: browse("model"),
    evModelBrowse: browse("model"),
    evFileBrowse: browse("held-out text file"),
    evTestFileBrowse: browse("test text file"),
    evResumeBrowse: browse("fine-tune checkpoint"),
    evJsonBrowse: browse("JSON output file", "save"),
    miPathBrowse: browse("model file"),
    ggPathBrowse: browse("GGUF file"),
    ggResumeBrowse: browse("fine-tune checkpoint"),
    ftPathBrowse: browse("GGUF file"),
    ftCorpusBrowse: browse("training corpus"),
    ftOutBrowse: browse("output checkpoint", "save"),
    ftResumeBrowse: browse("resume checkpoint"),
    exPathBrowse: browse("source GGUF"),
    exResumeBrowse: browse("fine-tune checkpoint"),
    exOutBrowse: browse("output GGUF file", "save"),
    svModelBrowse: browse("model to serve"),
  };

  const helpFor = (id) => HELP[id] || null;
  const isButton = (entry) => !!entry && typeof entry.does === "string";

  const api = { HELP, KINDS, helpFor, isButton };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumHelp = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
