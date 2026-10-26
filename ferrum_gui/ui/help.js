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
    "model_info", "gguf_info", "gguf_run", "gguf_finetune", "gguf_export", "capability",
  ];

  // A "Browse..." button next to a path field.
  const browse = (what, mode) => ({
    title: "Browse for " + what,
    does: mode === "save"
      ? "Opens the system 'save as' dialog and puts the file path you pick into the " + what + " field, so you do not have to type it."
      : mode === "dir"
        ? "Opens the system folder picker and puts the folder you choose into the " + what + " field, so you do not have to type it."
        : "Opens the system file picker and puts the path of the file you choose into the " + what + " field, so you do not have to type it.",
    kind: "instant",
    beginner: "A path is the address of a file on your computer, like /home/you/data/story.txt. Browsing just fills in that address for you; nothing is read or written until you press the action button.",
  });
  const tab = (name, what) => ({
    title: name + " tab",
    does: "Shows the " + name + " page: " + what,
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
      what: "The path of the training text file (the corpus). The model learns to continue text like this.",
      how: "The file must be non-empty and at least a little longer than the Context length in tokens, or training refuses to start. Use the Datasets tab to download and clean one. Its size also drives the time estimate.",
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
      how: "Reusing one keeps the vocabulary stable between experiments and resumes and skips tokenizer training. Not used for character-level models (Vocab 0). It must not be the same file as the model output.",
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
      how: "Works with models trained on the Train tab. The Train tab fills in the path after a run.",
      beginner: "Evaluating asks 'how surprised is the model by text it has never seen?'. Less surprise means a better model.",
    },
    evFile: {
      title: "Held-out text file",
      what: "A text file the model was not trained on, used to measure how well it generalises.",
      how: "If you also paste text in the box below, the pasted text is used and this file is ignored. The score is computed over every token with a sliding window; time is proportional to the text length.",
      beginner: "It is the exam paper. If the model saw these exact pages during training the score would be fake, so use text you set aside beforehand.",
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
      kind: "instant",
    },
    trStart: {
      title: "Train",
      does: "Trains a new transformer language model on the corpus with the settings above and saves it as a .gguf file. The loss chart and progress bar update live.",
      kind: "train",
      beginner: "Training is the slow part: the model reads the text many times, adjusting its numbers to predict the next token better. The time estimate comes from your CPU cores, the model size and the corpus size; it can be off by a factor of two either way.",
    },
    genStart: {
      title: "Generate",
      does: "Loads the model and writes new text that continues the seed, showing it live when streaming is on.",
      kind: "generate",
    },
    evRun: {
      title: "Evaluate and add row",
      does: "Scores the model on the held-out text and adds a row with perplexity, cross-entropy, bits per token and a verdict to the table.",
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
    evClear: {
      title: "Clear table",
      does: "Removes all rows from the evaluation table. No file is touched.",
      kind: "instant",
    },
    miInspect: {
      title: "Inspect / reload",
      does: "Reads the model file and shows its format, task, sizes, tokenizer and number of layers.",
      kind: "model_info",
    },
    ggInspect: {
      title: "Inspect",
      does: "Reads only the GGUF header and shows the architecture, shape, tokenizer and the RAM needed at int4, int8 and f32. Warns when the model exceeds this machine's estimated limits.",
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
      kind: "gguf_info",
    },
    exExport: {
      title: "Export",
      does: "Converts the model (with the fine-tune checkpoint applied, if given) to the chosen output type and writes a new GGUF file, with progress for each phase.",
      kind: "gguf_export",
    },
    capCheck: {
      title: "Check this machine",
      does: "Runs a short benchmark of memory speed and matrix-multiplication speed and estimates which model sizes this machine can load, train, fine-tune and run. In the release app the result also makes every time estimate in this window more accurate.",
      kind: "capability",
      beginner: "The numbers are rough upper bounds, not promises. Close other heavy programs for a fair reading.",
    },
    termClear: {
      title: "Clear",
      does: "Empties the Engine log panel. It does not affect any running job.",
      kind: "instant",
    },
    ggWarnCancel: {
      title: "Cancel",
      does: "Closes the warning and does not run the oversized model.",
      kind: "instant",
    },
    ggWarnProceed: {
      title: "Proceed anyway",
      does: "Continues despite the warning that the model exceeds this machine's estimated limits. The job may be very slow or fail.",
      kind: "instant",
    },
    capDialogClose: {
      title: "Close",
      does: "Closes the machine capability report. The numbers stay on the Capable tab.",
      kind: "instant",
    },

    // Tabs
    "tab-datasets": tab("Datasets", "download, load, clean and save the text your model learns from."),
    "tab-train": tab("Train", "train a new small transformer language model from scratch on your corpus."),
    "tab-generate": tab("Generate", "write new text with a trained model, optionally streamed live."),
    "tab-evaluate": tab("Evaluate", "score models on text they have not seen, using perplexity."),
    "tab-models": tab("Models", "inspect a model file."),
    "tab-gguf": tab("GGUF", "inspect and run an external Llama or Qwen model."),
    "tab-finetune": tab("Fine-tune", "adapt an external GGUF model to your own text."),
    "tab-export": tab("Export", "write a (fine-tuned) GGUF model to a new file in another number format."),
    "tab-system": tab("System", "live CPU, memory and thread load."),
    "tab-capable": tab("Capable", "benchmark this machine and see which model sizes it can handle."),

    // Browse buttons
    dsDestDirBrowse: browse("download folder", "dir"),
    dsFileBrowse: browse("local text file"),
    dsOutBrowse: browse("output corpus file", "save"),
    trCorpusBrowse: browse("training corpus"),
    trModelBrowse: browse("model output file", "save"),
    genModelBrowse: browse("model"),
    evModelBrowse: browse("model"),
    evFileBrowse: browse("held-out text file"),
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
