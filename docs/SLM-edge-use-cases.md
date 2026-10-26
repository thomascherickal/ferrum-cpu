# Small Is the New Big: 38 Real-World Jobs for Language Models Under 500M Parameters

## How Many Parameters Does Edge AI Really Need? (Far Fewer Than You Think!)

AI needs a data center.

AI needs a GPU that costs more than your car.

AI is useless below a billion parameters.

Every engineer building for phones, wearables and embedded boards has heard these comments.

I'm writing this article to tell you they could not be more wrong.

There are models running right now on $8 microcontrollers, on smart TVs with 3GB of RAM, inside your browser tab and on the keyboard you type on every day.

Some of them have fewer than a million parameters.

And they are doing real work.

Read this article to the end to find out why.

---

## But First - What the Heck Is a Small Language Model Anyway?

You know TinyML.

Keyword spotting on a Cortex-M chip. Anomaly detection on a vibration sensor. Models measured in kilobytes.

A small language model (SLM) is the same idea, scaled up just enough to understand language, speech, images and actions.

Not billions of parameters.

Millions. Sometimes hundreds of millions.

Running on the device itself. No cloud. No network round trip. No sending your private data to anyone.

And here is the rule I have taken from every source in this article:

**The narrower the task, the smaller the model can be. And fine-tuning is the multiplier.**

Every number below is tied to a model card, a paper or an engineering blog, linked inline and listed in full in the References at the end. I have not invented a single use case. If I could not verify it, it is not here.

---

## So How Many Parameters Do You Actually Need?

What is the honest answer? It depends on the job.

| Tier | Typical hardware | What works | Verified examples |
|---|---|---|---|
| Kilobytes to 30M | Microcontrollers, ESP32-S3, phone CPUs, browsers | Wake words, voice activity detection, keyboard prediction, grammar fixes, toxicity filters, translation, guardrails, speech recognition | microWakeWord (~22 KB arena), Gboard CIFG (1.4M), Firefox Translations (15.7M), Prompt Guard 2 (22M), Moonshine Tiny (27M) |
| 30M to 130M | Single-board computers, phones | Sensor reasoning, TTS, PII detection, phishing detection, robot policies, voice moderation | TinyLLM (30M to 124M), Whisper Tiny (39M), DistilBERT (66M), Kokoro (82M), Octo (27M and 93M) |
| 135M to 360M | Phones, Jetson-class boards, smart TVs | Function calling, narrow chat, extraction, embeddings, customer support | SmolLM2 (135M and 360M), Gemma 3 270M, FunctionGemma (270M), EmbeddingGemma (308M), LFM2-350M |
| 250M to 500M | Phones, Raspberry Pi 5, Jetson, laptops | Vision-language, documents, robot control, code, smart-home agents | SmolVLM (256M), SmolDocling (256M), Florence-2-base (0.23B), SmolVLA (450M), Qwen2.5-0.5B (0.49B) |

Three pieces of evidence set the floor and the ceiling.

**The floor is real.** In the NUS [TinyLLM study](https://arxiv.org/abs/2412.15304), the 30M and 50M models performed poorly on gesture detection, and the custom 30M model scored zero accuracy on that task. Larger custom models up to 124M did well.

**The ceiling is real too.** A study that put 21 quantized SLMs on a commercial [Android TV with an ARM Cortex-A55 and 3GB of RAM](https://sol.sbc.org.br/index.php/sbcup/article/download/43294/43060/) found most models above 500 million parameters unsuitable. Gemma-3-1B and Llama-3.2-1B crashed on load.

**Fine-tuning moves the needle more than size.** Google fine-tuned [FunctionGemma](https://blog.google/technology/developers/functiongemma/), a 270M model, and its Mobile Actions accuracy jumped from 58% to 85%.

One more thing.

Parameter counts flatter these models. [Gemma 3 270M](https://developers.googleblog.com/introducing-gemma-3-270m/) has 170M embedding parameters and only 100M in its transformer blocks. The compute is lighter than the headline number.

Now let's get to work.

---

## Tier 1: The TinyML Floor (Kilobytes to 30M)

### Use Case 1: Wake Words on a Microcontroller

Home Assistant states plainly that [openWakeWord is too large for low-power devices like the S3-BOX-3](https://www.home-assistant.io/voice_control/about_wake_word), so ESP32-S3 devices and Android phones run the lighter microWakeWord locally instead.

How small? The [ESPHome manifest for "okay nabu"](https://esphome.io/components/micro_wake_word/) asks for a tensor arena of 22,860 bytes.

About 22 KB.

Your phone's lock-screen wallpaper is bigger than that.

Want to build your own? One open project runs an [INT8 keyword spotter on an Arduino Nano 33 BLE Sense](https://github.com/HuyPopper-tech/TinyML-Voice-Control-Arduino) - a 64 MHz Cortex-M4F - switching LEDs and fans by voice in about 161 KB of flash, model and program together. No internet connection at all.

### Use Case 2: Voice Activity Detection

[Silero VAD](https://github.com/snakers4/silero-vad) is about 2MB. It processes a 30ms audio chunk in under 1ms on a single CPU thread, and lists IoT, edge and mobile as typical uses.

Every offline voice assistant starts here. Know when someone is speaking, and only then wake the bigger models.

### Use Case 3: Keyboard Next-Word Prediction

This one has been in your pocket since 2018.

Google trained [Gboard's next-word model](https://arxiv.org/abs/1811.03604) across phones with federated learning. A CIFG LSTM with about 1.4 million parameters, just 1.4MB after quantization.

And it beat the server-trained version on top-1 recall over client caches - 15.8% against 15.0%.

1.4 million parameters.

On billions of devices.

### Use Case 4: Grammar Correction As You Type

Gboard on the Pixel 6 [fixes grammar entirely on-device](https://research.google/blog/grammar-correction-as-you-type-on-pixel-6/) with a model of about 20MB - a Transformer encoder with an LSTM decoder, distilled from a large cloud grammar model.

Latency? Under 22ms for 60 characters on the Pixel 6 CPU.

Faster than you can blink.

### Use Case 5: Toxic Content Detection

The [Tiny-toxic-detector](https://arxiv.org/abs/2409.02114) has about 2.1 million parameters. Its authors report 90.97% accuracy on ToxiGen and 86.98% on Jigsaw, rivalling models more than 50 times its size, and they built it for resource-constrained moderation.

### Use Case 6: Browser Translation That Never Leaves Your Laptop

Firefox translates web pages locally. The [student models Mozilla ships](https://hacks.mozilla.org/2022/06/training-efficient-neural-network-models-for-firefox-translations/) have about 15.7M parameters and weigh roughly 17MB, distilled from a 192.75M-parameter teacher. They run on your CPU through WebAssembly.

Your reading habits stay yours.

### Use Case 7: Prompt-Injection Guardrails

Running an AI agent at the edge? Guard the front door.

Meta's [Llama Prompt Guard 2 22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M), built on DeBERTa-xsmall, flags prompt injections and jailbreaks. Meta reports 19.3ms per classification against 92.4ms for the 86M version, and a 75% cut in latency and compute for a small loss in accuracy. (The 22M figure is the backbone; Hugging Face lists 70.8M including embeddings.)

### Use Case 8: Lightweight Semantic Search

[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) has 22.7M parameters and produces 384-dimensional vectors for semantic search, clustering and sentence similarity.

Small enough to index a phone's notes, a field technician's manuals, or a kiosk's FAQ - entirely offline.

### Use Case 9: Speech Recognition in 27M Parameters

[Moonshine Tiny](https://arxiv.org/abs/2509.02523) has 27.1M parameters against 37.8M for Whisper Tiny. The monolingual versions average 48% lower error rates than Whisper Tiny and beat the 9x larger Whisper Small. Releases cover Arabic, Chinese, Japanese, Korean, Ukrainian and Vietnamese.

The [8-bit package](https://huggingface.co/OpenASR/moonshine-tiny) is about 34MB.

### Use Case 10: Classic On-Device NLP

[MobileBERT](https://arxiv.org/abs/2004.02984) has 25.3M parameters. It is 4.3x smaller and 5.5x faster than BERT-Base, lands within 0.6 points of it on GLUE, and runs in 62ms on a Pixel 4.

On a Snapdragon 8 Elite NPU, [Qualcomm lists it](https://huggingface.co/qualcomm/Mobile-Bert-Uncased-Google) at about 15ms.

### Use Case 11: Instant Replies on Smartwatches and Smart Glasses

Here is a clever one.

Researchers built [micro language models of 8M to 30M parameters](https://arxiv.org/abs/2604.19642) for watches and glasses that cannot run even 100M models continuously. The micro model writes the first 4 to 8 words instantly. A cloud model finishes the sentence.

The authors report these micro models match several existing models in the 70M to 256M range.

The user sees an answer start immediately.

Nobody waits.

### Use Case 12: Time-Series Forecasting for IoT

Not every edge model speaks English. Some speak sensor.

IBM's [TinyTimeMixer](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1) has fewer than one million parameters (805K) and runs on CPU-only machines. IBM says it beats forecasting benchmarks that demand billions of parameters, naming TimesFM, Moirai and Chronos.

Amazon's [Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-tiny) comes in 9M, 21M, 48M and 205M sizes, up to 250 times faster and 20 times more memory-efficient than the original Chronos. It deploys on CPU.

Energy meters. Cold-chain trucks. Factory sensors.

Forecasting at the edge - easy.

### Use Case 13: Text Generation on an $8 Chip

Can a microcontroller generate language at all?

A [260K-parameter TinyStories model](https://github.com/jstephencorey/esp32-llm) runs on an ESP32-S3 with 2MB of PSRAM at about 19 tokens per second. An [optimized fork](https://github.com/eric-humane/esp32-llm) reaches about 33.

And in 2026 a developer ran a [28.9M-parameter model on an ESP32-S3](https://circuitdigest.com/news/esp32-s3-runs-a-289-million-parameter-llm-borrowing-a-memory-trick-from-googles-gemma) with just 512KB of SRAM, streaming the 25M-parameter embedding table from flash at about 9 tokens per second.

I'll be honest with you. It writes short stories. It cannot answer questions or follow instructions.

But a 100x jump in two years?

Watch this space.

---

## Tier 2: Single-Task Specialists (30M to 130M)

### Use Case 14: Sensor Analytics on Single-Board Computers

The [TinyLLM framework](https://arxiv.org/abs/2412.15304) trains GPT-2-style models from 30M to 124M parameters, needing about 95MB to 313MB of RAM.

The tasks? Hand-gesture detection from a light sensor. Indoor localization from sensors on a moving robot. Swimming-style detection from smartwatch accelerometers.

The custom models matched or beat fine-tuned Phi and Llama models. On localization, the custom model finished more than 70 times faster. On an Orange Pi Zero 2W, where the larger models could not even load, it still produced about 6 tokens per second.

Gesture fine-tuning needed just 440 examples.

### Use Case 15: A Fully Offline Voice Assistant

[Whisper Tiny](https://github.com/openai/whisper) has 39M parameters.

On a [Turing Pi RK1 board (RK3588)](https://turingpi.com/whisper-cpp-piper-tts-arm64-turing-pi-rk3588/), Whisper Tiny transcribed a 5-minute clip in 29.79 seconds - a real-time factor of 0.10 at 299MB peak memory. Piper's 61MB "amy-medium" voice spoke back about 10 times faster than real time.

Once the models are downloaded, no internet is needed.

Listen, think, speak - all on a board in your cupboard.

### Use Case 16: Natural Text-to-Speech on a Phone

[Kokoro](https://soniqo.audio/guides/kokoro) has 82M parameters, built on StyleTTS 2 with an iSTFTNet vocoder at 24 kHz. Its Core ML port runs on the Apple Neural Engine at about 80MB in INT8.

NVIDIA ships a [Kokoro-82M TTS service](https://docs.nvidia.com/nemo/labs-voice-agent/nemo-voice-agent/nemo_voice_agent/pipecat/services/nemo/tts/) in its NeMo voice-agent framework.

### Use Case 17: Phishing and Smishing Detection

[DistilBERT](https://arxiv.org/abs/2402.13871) has 66M parameters - 40% smaller than BERT, keeping about 97% of its language understanding.

One project runs an [8-bit DistilBERT (about 67MB) entirely in the browser](https://v3x.is-a.dev/projects/phish/) through WebAssembly, so your text messages and 2FA codes never leave your device. A fine-tuned [DistilBERT URL classifier](https://huggingface.co/najlajj453/malicious-url-distilbert) reports an F1 of about 0.96 on phishing links.

### Use Case 18: Clinical PII Detection on Android

[OpenMed](https://huggingface.co/OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android) packages PII detectors at 33M, 66M, 110M, 335M and [340M](https://huggingface.co/OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-onnx-android) parameters as ONNX models for Android. They cover more than 55 identifier types across 17 languages, with tokenization and inference staying on the device.

They detect identifiers. They do not diagnose. Know the difference.

### Use Case 19: Voice Chat Moderation for Games

Roblox open-sourced a [voice safety classifier](https://huggingface.co/Roblox/voice-safety-classifier) fine-tuned from WavLM base plus - 94.6M parameters - trained on 2,374 hours of voice chat across six categories, including profanity and bullying.

Arm then ran it [on a phone CPU](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/on-device-toxic-speech-detection-with-ml-and-sme2). With static INT8 quantization and SME2 on a MediaTek Dimensity 9500, 15 seconds of audio took 0.37 seconds on a single thread.

### Use Case 20: Generalist Robot Policies

[Octo](https://octo-models.github.io/) comes in two sizes: about 27M (Small) and 93M (Base). It was [trained on about 800,000 trajectories](https://arxiv.org/abs/2405.12213) from the Open X-Embodiment dataset, evaluated across nine robot platforms, and fine-tunes to new robots in a few hours on standard consumer GPUs.

### Use Case 21: Automotive Translation

[Opus-MT](https://aihub.qualcomm.com/automotive/models/opus_mt_zh_en) has about 74M encoder and 74M decoder parameters. Qualcomm's AI Hub lists it on automotive chipsets including the SA7255P, SA8295P and SA8775P.

Road trip abroad, no signal?

No problem.

### Use Case 22: Financial Sentiment Analysis

[FinBERT](https://arxiv.org/abs/1908.10063) is a BERT-base model of about 110M parameters that beat prior state of the art on two financial sentiment datasets. [Follow-up work](https://arxiv.org/abs/2412.09859) matched state of the art by fine-tuning only about 15M of those 110M parameters.

Air-gapped trading desks will love this.

---

## Tier 3: Narrow Instruction Followers (135M to 360M)

### Use Case 23: Phone Control Agents

"Turn on the flashlight." "Create a calendar event." "Add a contact."

[FunctionGemma](https://blog.google/technology/developers/functiongemma/) (270M) turns requests like these into phone function calls, fully offline. Google positions it for phones and the Jetson Nano, as a standalone agent or as a router that hands harder requests to bigger models. Its 256K vocabulary tokenizes JSON efficiently, as [InfoQ notes](https://infoq.com/news/2026/01/functiongemma-edge-function-call).

The battery cost? Google's internal tests show the INT4 [Gemma 3 270M](https://developers.googleblog.com/introducing-gemma-3-270m/) used 0.75% of a Pixel 9 Pro's battery across 25 conversations.

Meta's [MobileLLM-350M](https://arxiv.org/abs/2402.14905) scored close to LLaMA-v2 7B on exact-match API calling. Meta estimates a 350M 8-bit model at 0.035 joules per token - enough for a full day of conversational use on an iPhone.

20 times smaller than LLaMA-v2 7B.

Roughly the same answer.

### Use Case 24: Voice-Controlled Games

Google's TinyGarden demo, on the [same FunctionGemma page](https://blog.google/technology/developers/functiongemma/), lets players say "Plant sunflowers in the top row and water them." The 270M model breaks that into plantCrop and waterCrop calls on specific grid cells, handling multi-turn logic on the phone without ever touching a server.

### Use Case 25: Customer Support on a Smart TV

Remember the Android TV study? Here is what actually worked.

Nine of the 21 models ran usably on that [3GB, Cortex-A55 TV](https://sol.sbc.org.br/index.php/sbcup/article/download/43294/43060/), including Lite Oute1-65M, SmolLM2-135M, SmolLM2-360M, Pythia410m-Instruct and OpenELM-450M-Instruct. The test queries were real TV support problems - HDMI not working, connecting the antenna, opening Netflix, finding the date settings.

And the runtime matters. Switching SmolLM2-360M from llama.cpp to MNN lifted decode speed from 1.60 to 19.34 tokens per second.

About 12 times faster.

Same model. Same TV.

### Use Case 26: On-Device Chat, Rewriting and Extraction

Hugging Face built [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) at 135M and 360M to run on-device. Quantized, the [135M model is about 86MB and the 360M about 256MB](https://hub.docker.com/r/ai/smollm2).

Liquid AI's [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) runs on CPU, GPU and NPU for smartphones, laptops and vehicles, aimed at agentic tasks, extraction and RAG. [LM Studio](https://lmstudio.ai/models/lfm2) lists about 220MB of RAM for the smallest LFM2. Liquid itself advises against knowledge-heavy or coding tasks.

I respect that honesty.

### Use Case 27: Private Search and RAG

[EmbeddingGemma](https://developers.googleblog.com/en/introducing-embeddinggemma/) has 308M parameters - roughly 100M model and 200M embedding parameters. It runs in [under 200MB of RAM when quantized](https://deepmind.google/models/gemma/embeddinggemma/), supports more than 100 languages, and Google DeepMind reports it as the top open multilingual embedding model under 500M on MTEB.

Hugging Face shows how to [fine-tune it on the MIRIAD medical retrieval dataset](https://huggingface.co/blog/embeddinggemma).

Your documents. Your device. Your business.

### Use Case 28: Named-Entity Extraction for Any Entity Type

[GLiNER](https://arxiv.org/abs/2311.08526) extracts any entity type you name, and its authors report it beats ChatGPT and fine-tuned LLMs in zero-shot NER. The [small version](https://huggingface.co/urchade/gliner_small-v2.1) has 166M parameters, the medium 209M and the large 459M, all Apache-2.0.

Invoices, contracts, field reports - extraction on-device - all in a day's work.

---

## Tier 4: Multimodal and Code (250M to 500M)

### Use Case 29: Edge Vision, Even in the Browser

[SmolVLM-256M](https://arxiv.org/abs/2504.05299) uses under 1GB of GPU memory and outperforms Idefics-80B, a model 300 times larger.

You can run it [inside a browser tab with Transformers.js and WebGPU](https://pyimagesearch.com/2025/10/20/running-smolvlm-locally-in-your-browser-with-transformers-js/), with images and text staying on your device.

### Use Case 30: Captioning, Detection and OCR in One Model

Microsoft's [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) has 0.23B parameters and handles captioning at three detail levels, object detection, phrase grounding, dense region captioning and OCR, all switched by a text prompt.

One model. Seven jobs.

### Use Case 31: Document Conversion and OCR

[SmolDocling](https://arxiv.org/abs/2503.11576) is a 256M vision-language model from IBM Research that converts whole document pages - code, tables, equations, charts - and its authors say it competes with models up to 27 times larger. A [GGUF build](https://huggingface.co/cstr/smoldocling-GGUF) is 153MB at Q4_K.

Need classic text recognition on a phone? PaddleOCR lists its mobile detection, recognition and direction models at [about 9.4M in total](https://www.paddleocr.ai/latest/en/version2.x/ppocr/overview.html), with deployment on iOS and Android.

### Use Case 32: Accessibility for Blind and Low-Vision Users

This one matters to me deeply.

[Pipecat's SmolVLM service](https://docs.pipecat.ai/api-reference/server/services/vision/smolvlm.md) defaults to SmolVLM-256M for on-device image description. Pair it with Kokoro (82M) for speech, and you have local scene description that never sends a picture of someone's home to a server.

IIT Mandi researchers [tested SmolVLM2 at 500M and 2.2B on a smartphone](https://arxiv.org/abs/2511.10615) for blind and low-vision description. A 2026 follow-up, [Smol-VL-BLV](https://arxiv.org/pdf/2609.28757), adds hazard and direction awareness to a 500M model that fits in about 450MB and runs fully offline.

To be precise: those accessibility studies sit right at 500M, not under it. The 256M pairing is the strictly sub-500M route.

By God's grace, a billion people with vision impairment could soon carry a describer in their pocket.

### Use Case 33: Robots That Follow Instructions

[SmolVLA](https://huggingface.co/blog/smolvla) is a 450M vision-language-action model with an action expert of about 100M parameters. Pretraining on [10 million frames from 487 community datasets](https://pureai.com/articles/2025/06/10/hugging-face-releases-smolvla.aspx) lifted success on the SO100 tasks from 51.7% to 78.3%, and it runs on CPUs and single consumer GPUs, MacBooks included.

TU Wien [fine-tuned it for the 11-joint Toyota HSR](https://arxiv.org/pdf/2606.00253) mobile manipulator and compared it with the 3.3B π0.5.

### Use Case 34: A Smart-Home Brain on a Raspberry Pi

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) has 0.49B parameters, 0.36B of them non-embedding.

In the open-source [openHome project](https://huggingface.co/Xx-Vexento-xX/openhome-nova), ESP32 sensors report to a Raspberry Pi hub where NOVA - Qwen2.5-0.5B fine-tuned on about 1,900 examples - reasons over events and voice commands through Ollama on a Pi 3B+ or newer.

A hobbyist build, not a product. But it proves the point.

### Use Case 35: Clinical De-identification

[EdgeDeID](https://siim.org/wp-content/uploads/2025/06/EdgeDeID-Advancing-De-identification-with-Small-LMs-Bardia-Khosravi.pdf) fine-tuned Qwen2.5-Coder-0.5B on synthetic medical reports, generated by a 405B model, to strip patient identifiers on edge devices.

The giant writes the training data. The small model does the daily work.

### Use Case 36: Agriculture in Low-Connectivity Areas

[Farm-LightSeek](https://arxiv.org/abs/2506.03168) rebuilt LLaVA around Qwen2.5-0.5B to analyse agricultural IoT images and data at the edge, distilling capability from a larger teacher.

Closer to home, India's [KisanSLM](https://aikosh.indiaai.gov.in/home/models/details/kisan_chat.html) is fine-tuned on Kisan Call Centre transcripts for multilingual offline farm advice. Its listing does not publish a parameter count, so I cannot place it in a tier.

### Use Case 37: Offline Code Completion

[Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) brings code assistance to laptops and small boards with no cloud dependency. Your proprietary code stays on your machine.

### Use Case 38: Domain-Tuned SLMs for Healthcare, Finance and Legal

The [Shakti family](https://arxiv.org/abs/2503.01933) - including 100M and 250M variants under our line - was benchmarked on healthcare, finance and legal tasks for smartphones, smart appliances and IoT systems.

---

## What These Models Can Already Do

Let me put it all in one place.

- Detect a wake word in about 22 KB of working memory - piece of cake.
- Predict your next word with 1.4 million parameters - easy.
- Fix your grammar in under 22 milliseconds - done!
- Translate a web page with a 17MB model inside your browser - sure thing!
- Transcribe 5 minutes of speech in 30 seconds on an ARM board - no problem.
- Turn "plant sunflowers and water them" into function calls on a phone - all in a day's work.
- Convert a scanned page with tables and equations using 256M parameters - doable.
- Drive a robot arm from camera input with 450M parameters - possible.

I could go on and on.

Actually - I just did. Thirty-eight times.

---

## But Thomas - Can a 135M Model Actually Think?

Now you can say, wait a minute, Thomas.

These are narrow tools. They follow one script. A 135M model cannot reason through a biology exam. It cannot plan a drone mission. It cannot run a satellite. Aren't you overselling this?

True.

But that was never the claim.

The claim is that most edge jobs are narrow, and narrow jobs do not need giants.

Here is the honest list of where the giants are still needed.

## Now; for the Other Side of the Coin!

| Domain | What the evidence shows | Sizes involved |
|---|---|---|
| Education and tutoring | An offline AI tutor study found SmolLM2-135M scored about 21.8% on four-choice MMLU biology questions with retrieval, below random guessing, while SmolLM2-1.7B reached 33% to 41%. [Source](https://arxiv.org/abs/2510.06255) Uganda's [Arapai](https://arxiv.org/abs/2603.03339) runs quantized models on CPU-only machines for 120 students, without publishing model sizes. | 1.7B needed for tutoring quality |
| Drones | A PX4 drone framework controlled by natural language used models from [3B to 12B](https://arxiv.org/abs/2506.07509), served from a workstation rather than onboard. | 3B+ |
| Space | Satlyt runs a quantized [Gemma 3 1B onboard a satellite](https://deepmind.google/models/gemma/gemmaverse/satlyt/) at about 19 tokens per second in ground tests. [KENSAT](https://blog.circuit.rocks/?p=6600) runs TinyLlama on a Jetson Orin Nano in a CubeSat. A payload processor ran [Gemma-3n (~3.8B) at about 15W](https://smallsateurope.com/paper/payload-operation-using-on-board-vision-language-model/). | 1B to 3.8B |
| Desktop writing assistance | Grammarly's [on-device corrector](https://www.grammarly.com/blog/engineering/efficient-on-device-writing-assistance/) is about 1B parameters, running at about 210 tokens per second on an M2 Mac. | ~1B |
| Wearable health prediction | [HealthSLM-Bench](https://neurips.cc/virtual/2025/124917) and a [mobile health prediction study](https://arxiv.org/abs/2409.18987) evaluate models mostly from 1.1B to 3.8B. | 1B+ |
| Agentic farm advice | A benchmark of 20 SLMs for [dairy-farm decision support](https://arxiv.org/abs/2512.14043) found models under 2B failed to follow instructions. | 2B+ |
| Industrial maintenance | Vendors recommend [sub-1B models on RK3576 and RK3588 NPUs](https://www.geniatech.com/edge-ai-predictive-maintenance-guide/), and [Syn-Diag](https://arxiv.org/abs/2510.05733) distils a fault-diagnosis model into an edge version 83% smaller. Neither publishes exact parameter counts. | Unverified |

Look at the space row again.

Satlyt is targeting a memory reduction of more than 90% through pruning and distillation for some diagnostic workloads.

That is a 1B model heading toward the sub-500M tier.

The line keeps moving down.

---

## So How Do You Pick the Right Size?

1. **Define the task brutally narrowly.** Classification, transcription or a fixed set of function calls can live under 130M. Open-ended tutoring cannot.
2. **Start from a specialist.** FunctionGemma for tools, EmbeddingGemma for retrieval, Moonshine for speech, SmolDocling for documents. A specialist beats a generalist of the same size.
3. **Budget for fine-tuning.** TinyLLM needed 440 gesture examples. NOVA used about 1,900. That is an afternoon of data work, not a research grant.
4. **Pick the runtime as carefully as the model.** MNN made the same model 12 times faster on the same TV.
5. **Plan the escape hatch.** Route hard requests to a larger model, exactly as Google suggests for FunctionGemma and as the micro-model researchers do on watches and glasses.

---

## Where Will You Be?

The giants will keep getting bigger. That's fine.

But the real revolution is happening in your pocket, on your wrist, in your car, in your browser and in a $8 chip on a workbench somewhere in Chennai.

Where will you be when every device you own thinks for itself?

Build small.

Fine-tune.

Quantize.

Measure.

Ship.

Somewhere tonight, a 1.4-million-parameter LSTM is predicting the next word for someone typing to their mother, and it fits in less space than a single photo.

That is the future.

And the future is now.

Cheers!

---

## References

1. TinyML Voice Control on Arduino Nano 33 BLE Sense - https://github.com/HuyPopper-tech/TinyML-Voice-Control-Arduino
2. Home Assistant: The approach to wake words - https://www.home-assistant.io/voice_control/about_wake_word
3. ESPHome: Micro Wake Word - https://esphome.io/components/micro_wake_word/
4. Silero VAD - https://github.com/snakers4/silero-vad
5. Federated Learning for Mobile Keyboard Prediction (Gboard) - https://arxiv.org/abs/1811.03604
6. Grammar Correction as You Type, on Pixel 6 (Google Research) - https://research.google/blog/grammar-correction-as-you-type-on-pixel-6/
7. Tiny-toxic-detector - https://arxiv.org/abs/2409.02114
8. Training efficient neural network models for Firefox Translations (Mozilla Hacks) - https://hacks.mozilla.org/2022/06/training-efficient-neural-network-models-for-firefox-translations/
9. Llama Prompt Guard 2 22M model card - https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M
10. all-MiniLM-L6-v2 model card - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
11. Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices - https://arxiv.org/abs/2509.02523
12. Moonshine Tiny (OpenASR) - https://huggingface.co/OpenASR/moonshine-tiny
13. MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices - https://arxiv.org/abs/2004.02984
14. MobileBERT optimized for Qualcomm devices - https://huggingface.co/qualcomm/Mobile-Bert-Uncased-Google
15. Micro Language Models Enable Instant Responses - https://arxiv.org/abs/2604.19642
16. IBM Granite TinyTimeMixer (TTM) R1 - https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1
17. Amazon Chronos-Bolt Tiny - https://huggingface.co/amazon/chronos-bolt-tiny
18. Running an LLM on the ESP32 (jstephencorey) - https://github.com/jstephencorey/esp32-llm
19. Running an LLM on the ESP32-S3 (eric-humane) - https://github.com/eric-humane/esp32-llm
20. ESP32-S3 Runs a 28.9-Million-Parameter LLM (CircuitDigest) - https://circuitdigest.com/news/esp32-s3-runs-a-289-million-parameter-llm-borrowing-a-memory-trick-from-googles-gemma
21. TinyLLM: A Framework for Training and Deploying Language Models at the Edge - https://arxiv.org/abs/2412.15304
22. OpenAI Whisper - https://github.com/openai/whisper
23. Whisper.cpp and Piper TTS on the Turing Pi RK1 (RK3588) - https://turingpi.com/whisper-cpp-piper-tts-arm64-turing-pi-rk3588/
24. Kokoro TTS on-device guide - https://soniqo.audio/guides/kokoro
25. NVIDIA NeMo Voice Agent TTS services - https://docs.nvidia.com/nemo/labs-voice-agent/nemo-voice-agent/nemo_voice_agent/pipecat/services/nemo/tts/
26. An Explainable Transformer-based Model for Phishing Email Detection - https://arxiv.org/abs/2402.13871
27. Phish: client-side DistilBERT smishing detector - https://v3x.is-a.dev/projects/phish/
28. Malicious URL Detection with DistilBERT - https://huggingface.co/najlajj453/malicious-url-distilbert
29. OpenMed PII Detection 33M (Android) - https://huggingface.co/OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android
30. OpenMed PII Detection 340M (Android) - https://huggingface.co/OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-onnx-android
31. Roblox Voice Safety Classifier - https://huggingface.co/Roblox/voice-safety-classifier
32. On-device toxic speech detection with ML and SME2 (Arm) - https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/on-device-toxic-speech-detection-with-ml-and-sme2
33. Octo: An Open-Source Generalist Robot Policy (project page) - https://octo-models.github.io/
34. Octo: An Open-Source Generalist Robot Policy (paper) - https://arxiv.org/abs/2405.12213
35. Opus-MT on automotive chipsets (Qualcomm AI Hub) - https://aihub.qualcomm.com/automotive/models/opus_mt_zh_en
36. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models - https://arxiv.org/abs/1908.10063
37. Financial Sentiment Analysis: Leveraging Actual and Synthetic Data - https://arxiv.org/abs/2412.09859
38. FunctionGemma: Bringing bespoke function calling to the edge (Google) - https://blog.google/technology/developers/functiongemma/
39. Google Releases Gemma 3 270M Variant Optimized for Function Calling (InfoQ) - https://infoq.com/news/2026/01/functiongemma-edge-function-call
40. Introducing Gemma 3 270M (Google Developers Blog) - https://developers.googleblog.com/introducing-gemma-3-270m/
41. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases - https://arxiv.org/abs/2402.14905
42. Deploying Language Models on Android-Based Edge Devices (SBCUP) - https://sol.sbc.org.br/index.php/sbcup/article/download/43294/43060/
43. SmolLM2-360M model card - https://huggingface.co/HuggingFaceTB/SmolLM2-360M
44. SmolLM2 quantized builds (Docker Hub) - https://hub.docker.com/r/ai/smollm2
45. LFM2-350M model card (Liquid AI) - https://huggingface.co/LiquidAI/LFM2-350M
46. LFM2 (LM Studio) - https://lmstudio.ai/models/lfm2
47. Introducing EmbeddingGemma (Google Developers Blog) - https://developers.googleblog.com/en/introducing-embeddinggemma/
48. EmbeddingGemma (Google DeepMind) - https://deepmind.google/models/gemma/embeddinggemma/
49. Welcome EmbeddingGemma (Hugging Face Blog) - https://huggingface.co/blog/embeddinggemma
50. GLiNER: Generalist Model for Named Entity Recognition - https://arxiv.org/abs/2311.08526
51. GLiNER small v2.1 model card - https://huggingface.co/urchade/gliner_small-v2.1
52. SmolVLM: Redefining small and efficient multimodal models - https://arxiv.org/abs/2504.05299
53. Running SmolVLM Locally in Your Browser with Transformers.js (PyImageSearch) - https://pyimagesearch.com/2025/10/20/running-smolvlm-locally-in-your-browser-with-transformers-js/
54. Florence-2-base model card (Microsoft) - https://huggingface.co/microsoft/Florence-2-base
55. SmolDocling: An ultra-compact vision-language model for end-to-end document conversion - https://arxiv.org/abs/2503.11576
56. SmolDocling-256M GGUF - https://huggingface.co/cstr/smoldocling-GGUF
57. PaddleOCR PP-OCR overview - https://www.paddleocr.ai/latest/en/version2.x/ppocr/overview.html
58. Pipecat SmolVLM local vision service - https://docs.pipecat.ai/api-reference/server/services/vision/smolvlm.md
59. Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals - https://arxiv.org/abs/2511.10615
60. Small yet Assistive: Spatially-Aware Post-Training for Low Vision - https://arxiv.org/pdf/2609.28757
61. SmolVLA (Hugging Face Blog) - https://huggingface.co/blog/smolvla
62. Hugging Face Releases SmolVLA (Pure AI) - https://pureai.com/articles/2025/06/10/hugging-face-releases-smolvla.aspx
63. Fine-Tuning VLA Models for 11-DoF Mobile Manipulation (TU Wien) - https://arxiv.org/pdf/2606.00253
64. Qwen2.5-0.5B-Instruct model card - https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
65. openHome NOVA - https://huggingface.co/Xx-Vexento-xX/openhome-nova
66. EdgeDeID: De-identification with Small LMs on Edge Devices (SIIM) - https://siim.org/wp-content/uploads/2025/06/EdgeDeID-Advancing-De-identification-with-Small-LMs-Bardia-Khosravi.pdf
67. Farm-LightSeek: Edge-centric Multimodal Agricultural IoT Data Analytics - https://arxiv.org/abs/2506.03168
68. KisanSLM (IndiaAI AIKosh) - https://aikosh.indiaai.gov.in/home/models/details/kisan_chat.html
69. Qwen2.5-Coder-0.5B-Instruct model card - https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
70. Fine-Tuning Small Language Models for Domain-Specific AI (Shakti) - https://arxiv.org/abs/2503.01933
71. Towards an Efficient, Customizable, and Accessible AI Tutor - https://arxiv.org/abs/2510.06255
72. Arapai: Offline-first AI chatbot for low-connectivity education - https://arxiv.org/abs/2603.03339
73. Taking Flight with Dialogue: Natural Language Control for PX4-based Drones - https://arxiv.org/abs/2506.07509
74. How Satlyt is building local AI for space with Gemma (Google DeepMind) - https://deepmind.google/models/gemma/gemmaverse/satlyt/
75. KENSAT: A Home-Built CubeSat Running a TinyLlama LLM in Orbit - https://blog.circuit.rocks/?p=6600
76. Payload Operation using On-Board Vision Language Model (SmallSat Europe) - https://smallsateurope.com/paper/payload-operation-using-on-board-vision-language-model/
77. Efficient on-device writing assistance (Grammarly Engineering) - https://www.grammarly.com/blog/engineering/efficient-on-device-writing-assistance/
78. HealthSLM-Bench (NeurIPS 2025 workshop) - https://neurips.cc/virtual/2025/124917
79. Efficient and Personalized Mobile Health Event Prediction via Small Language Models - https://arxiv.org/abs/2409.18987
80. Evaluating Small Language Models for Agentic On-Farm Decision Support Systems - https://arxiv.org/abs/2512.14043
81. Edge AI Predictive Maintenance Guide (Geniatech) - https://www.geniatech.com/edge-ai-predictive-maintenance-guide/
82. Syn-Diag: Few-shot Fault Diagnosis on the Edge - https://arxiv.org/abs/2510.05733
