<p align="center"><img src="../../assets/logo.png" width="96" alt="Ferrum logo"></p>

# Ferrum SLM Studio: Beginner Mode & Advanced Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **One agent at a time** (CLAUDE.md working rule 2): subagent-driven means strictly sequential, never parallel.

**Goal:** Make `ferrum_gui` ("Ferrum SLM Studio") immediately approachable for first-time AI/SLM users by introducing a persistent **Beginner / Advanced** mode switch, model size presets (`Tiny`, `Small`, `Medium`, `Custom`), curated Beginner tabs and forms, guided "Next step" workflow calls-to-action, a collapsible Engine log dock, and mode-aware tooltips—while preserving 100 % of existing power-user controls in Advanced mode and keeping every existing HTML, CSP, accessibility, tooltip, and IPC fixture test green.

**Architecture:**
1. **CSS-driven progressive disclosure (`body[data-mode="beginner"]` vs. `body[data-mode="advanced"]`):** Advanced-only tabs, form rows, fieldsets, and table columns are tagged with `class="adv-only"` in `ferrum_gui/ui/index.html`. Hiding elements via CSS (`body[data-mode="beginner"] .adv-only { display: none !important; }`) instead of removing DOM nodes guarantees that every existing JS module (`*_tab.js`, `settings.js`, `live.js`) and every DOM guard test (`tooltips_guard.test.js`, `a11y_html.test.js`, `csp.test.js`, `app_flows.test.js`) continues to find all IDs in the DOM without null-reference checks.
2. **Dedicated `ui/core/mode.js` module:** Manages the mode state (`"beginner"` default for new installs, `"advanced"` toggleable in one click from the top header bar), persists it in `localStorage` alongside `settings.js`, auto-redirects to a visible tab if the user switches to Beginner mode while viewing an `.adv-only` tab, and controls the collapsible `#terminal` dock.
3. **Model Size Presets (`#trPreset`):** Bridges `slm_cli::preset` (`tiny`, `small`, `medium`) to the Train tab so beginners pick a human-readable size (`Tiny (~0.2M params — minutes)`, `Small (~1.5M params — hours)`, `Medium (~7M params — 1–2 days)`) instead of manually editing 7 interdependent architecture numbers or accidentally training the 16-token toy default.
4. **Guided Workflow CTAs:** Contextual banner buttons connect the core pipeline (`1. Datasets` → `2. Train` → `3. Generate` & `4. Evaluate`) and carry file paths forward automatically.
5. **Mode-aware Tooltips (`ui/core/tooltip.js`):** Renders concise, audience-tailored popovers—leading with `New to AI?` + `What it is` + live estimate in Beginner mode (omitting low-level `How it works` mechanics), and showing `What it is` + `How it works` + live estimate in Advanced mode.

**Tech Stack:** Vanilla ES modules (`ferrum_gui/ui/core/`, `ferrum_gui/ui/tabs/`), HTML5 + CSS3 custom properties (`ferrum_gui/ui/index.html`, `ferrum_gui/ui/styles.css`), Node 24 test suite (`ferrum_gui/ui/tests/*.test.js`), Rust / Tauri v2 (`ferrum_gui/src/`).

**Spec & Sources:**
- `docs/gap-report.md` (UX1–UX6 and H5 in the 2026-11-07 analysis)
- `docs/gui-tooltips.md` (tooltip structure and guard rules)
- `slm_cli/src/preset.rs` (`tiny`, `small`, `medium` preset definitions)

---

## UX Diagnosis: Why the Current GUI Overwhelms Beginners

| # | Current State in `ferrum_gui/ui/` | Impact on a Beginner | Beginner Mode Solution |
|---|---|---|---|
| **UX1** | **12 flat tabs** (`Datasets, Prepare, Train, Generate, Evaluate, Models, GGUF, Fine-tune, Export, Serve, System, Capable`) with identical visual weight. | A new user cannot tell where to start or which tabs belong to Ferrum's own GPT-2 models vs. imported Llama/Qwen GGUFs. | Show **6 tabs** in Beginner mode: numbered workflow steps `1. Datasets`, `2. Train`, `3. Generate`, `4. Evaluate`, plus `5. GGUF (Run External)` and `Capable`. Hide `Prepare`, `Models`, `Fine-tune`, `Export`, `Serve`, and `System` behind `.adv-only`. |
| **UX2** | **25+ raw controls on the Train tab**, including `Embed dim`, `Heads` (must divide `Embed dim`), `Blocks`, `Hidden`, `Vocab (0/≥256)`, `Stride`, `Token budget`, `Weight decay`, `Dropout`, `Grad clip`, `Warmup steps`, `Val fraction`, `Patience`, `Checkpoint every`, `Cosine LR`, `Tie embedding + head`, `Int8 QAT`, `Save precision`, and the `Tokenize a large corpus first` fieldset. | High cognitive load; changing `Embed dim` or `Heads` easily triggers the divisibility error before a user has trained their first model. | Show **4 inputs** in Beginner mode: `Corpus`, `Model output`, `Model size preset` (`Tiny` / `Small` / `Medium`), and `Epochs`, plus the `Pre-flight check` card and `Train` button. All 22 low-level hyperparameters are `.adv-only`. |
| **UX3** | **Toy defaults on the Train tab** (`context=16, embed=32, heads=4, blocks=2, hidden=64`). | Built for fast unit tests, not real text. A beginner who clicks `Train` without changing 7 numbers gets a 16-token toy model. | Default `#trPreset` to `Tiny (64 ctx, 64 dim, 4 heads, 2 blocks, 256 hidden, 512 vocab, lr 0.01)` on fresh Beginner-mode selection, or let the user pick `Tiny` / `Small` / `Medium` in one click. |
| **UX4** | **No hand-off between tabs.** Saving a corpus on `Datasets` or finishing `Train` prints a static status line; the user must manually find the next tab and re-type paths. | Friction between steps; beginners don't know what to do after `Saved 1.2 MB -> data/corpus.txt`. | Show a green **Next step** CTA bar: on `Datasets` → `Next: Train a model on this corpus →`; on `Train` completion → `Next: Generate text →` and `Evaluate model →`. |
| **UX5** | **Always-open 230px `#terminal` Engine log dock** at the bottom of the window (`grid-template-rows: 52px 1fr 230px`). | Consumes ~25 % of vertical screen space on laptops for `--verbose` logs that are off by default. | Collapse `#terminal` to a 32px header bar in Beginner mode (with a `Show log ▾` toggle button); auto-expand in Advanced mode or when `Verbose` is checked. |
| **UX6** | **4-section tooltip wall of text** (`What it is`, `How it works`, `New to AI?`, `With your current settings`, ~80–120 words). | Advanced users skim past the `New to AI?` analogy; beginners get bogged down in matrix/softmax mechanics under `How it works`. | `tooltip.js` checks `document.body.dataset.mode`: Beginner mode shows `New to AI?` + `What it is` + live estimate; Advanced mode shows `What it is` + `How it works` + live estimate. |

---

## Per-Tab Field Curation Matrix (Beginner vs. Advanced Mode)

Every `.adv-only` element remains in `index.html` and retains its ID, label, and `help.js` entry.

| Tab | Visible in Beginner Mode? | Controls Visible in Beginner Mode | Controls Tagged `.adv-only` (Hidden in Beginner Mode) |
|---|---|---|---|
| **1. Datasets** (`panel-datasets`) | **Yes** | Curated dataset table (`#datasetTable` 1-click download), `Source URL` (`#dsUrl`), `Local file / folder` (`#dsLocal` + Browse), `Clean & preview` (`#btnClean`), `Output file` (`#dsOut`), `Save corpus` (`#btnSaveCorpus`), plus new **Next: Train →** CTA banner. | `HuggingFace repo` row (`#dsHfRepo`, `#dsHfFile`, `#btnFetchHf`), `Kaggle dataset` row (`#dsKaggleSlug`, `#dsKaggleFile`, `#btnFetchKaggle`), `Cleaning options` fieldset (`#cleanHtml`, `#cleanUrls`, `#cleanWs`, `#cleanCtrl`, `#cleanDedup`, `#cleanMinLen`). |
| **Prepare** (`panel-prepare`) | **No** (`.adv-only` tab) | — (Beginners use Datasets → Clean & Save; deterministic `train/val/test` split + `manifest.json` is an Advanced MLOps workflow.) | Entire tab button `<button class="tab adv-only" data-tab="prepare">` |
| **2. Train** (`panel-train`) | **Yes** | `Corpus` (`#trCorpus` + Browse), `Model output` (`#trOut` + Browse), `Model size preset` (`#trPreset`: Tiny / Small / Medium / Custom), `Epochs` (`#trEpochs`), `Pre-flight check` card (`#trPreflight`), `Train` (`#btnTrain`), `Stop` (`#btnTrainCancel`), progress bar (`#trProgress`), plus **Next: Generate / Evaluate →** CTA banner. | `Tokenize a large corpus first` fieldset (`#tkInput`, `#tkOut`, `#tkVocab`, `#btnTokenize`, etc.), `Context length` (`#trContext`), `Embed dim` (`#trEmbed`), `Heads` (`#trHeads`), `Blocks` (`#trBlocks`), `Hidden` (`#trHidden`), `LR` (`#trLr`), `Batch` (`#trBatch`), `Vocab` (`#trVocab`), `Stride` (`#trStride`), `Token budget` (`#trBudget`), `Weight decay` (`#trDecay`), `Dropout` (`#trDropout`), `Grad clip` (`#trClip`), `Warmup steps` (`#trWarmup`), `Val fraction` (`#trValFrac`), `Patience` (`#trPatience`), `Threads` (`#trThreads`), `Seed` (`#trSeed`), `Tokenizer file` (`#trTokPath`), `Checkpoint file` (`#trCkptPath`), `Checkpoint every` (`#trCkptEvery`), `Resume` (`#trResume`), `Cosine LR` (`#trCosine`), `Tie embedding` (`#trTie`), `Int8 QAT` (`#trQat`), `Save precision` (`#trSaveQuant`), `Verbose` (`#trVerbose`). |
| **3. Generate** (`panel-generate`) | **Yes** | `Model` (`#genModel` + Browse), `Seed text` (`#genSeed`), `Chars` (`#genLen`), `Temperature (0 = greedy)` (`#genTemp`), `Stream output live` (`#genStream`), `Generate` (`#btnGenerate`), `Stop` (`#btnGenCancel`), output preview (`#genOutput`). | `Gen seed` (`#genRng`), `Max tokens` (`#genMaxTokens`), `Top-k` (`#genTopK`), `Top-p` (`#genTopP`), `Repetition penalty` (`#genRep`), `Stop strings` (`#genStop`), `Verbose` (`#genVerbose`). |
| **4. Evaluate** (`panel-evaluate`) | **Yes** | `Model (.gguf)` (`#evModel` + Browse), `Held-out text file` (`#evTextPath` + Browse), `Paste held-out text` (`#evTextInline`), `Evaluate & add row` (`#btnEval`), `Clear table` (`#btnEvalClear`), comparison table (`Model`, `Vocab`, `Tokens`, `Perplexity`, `Uniform (= V)`, `Verdict`). | `Test text file` row (`#evTestPath`, `#btnBrowseEvTest`), `Lossy` checkbox (`#evLossy`), `Imported Llama / Qwen GGUF only` fieldset (`#evQuant`, `#evResume`, `#evForce`), `Save JSON` (`#btnEvalSaveJson`), and table columns `Cross-entropy (nats)`, `Bits/token`, `Bits/byte`, `Test PPL` (`.adv-only` on `<th>`/`<td>`). |
| **Models** (`panel-models`) | **No** (`.adv-only` tab) | — | Entire tab button `<button class="tab adv-only" data-tab="models">` |
| **5. GGUF** (`panel-gguf`) | **Yes** | `GGUF file` (`#ggufPath` + Browse), `Inspect header & tensors` (`#btnGgufInspect`), `Prompt` (`#ggufPrompt`), `Max new tokens` (`#ggufMaxNew`), `Temperature` (`#ggufTemp`), `Run` (`#btnGgufRun`), `Stop` (`#btnGgufCancel`), output (`#ggufOutput`). | `Precision` (`#ggufQuant`), `Top-k` (`#ggufTopK`), `Top-p` (`#ggufTopP`), `Repetition penalty` (`#ggufRep`), `Gen seed` (`#ggufSeed`), `Stop strings` (`#ggufStop`), `Token IDs` (`#ggufIds`), `Fine-tune checkpoint` (`#ggufResume` + Browse), `Chat format` (`#ggufChat`), `Force` (`#ggufForce`). |
| **Fine-tune** (`panel-finetune`) | **No** (`.adv-only` tab) | — | Entire tab button `<button class="tab adv-only" data-tab="finetune">` |
| **Export** (`panel-export`) | **No** (`.adv-only` tab) | — | Entire tab button `<button class="tab adv-only" data-tab="export">` |
| **Serve** (`panel-serve`) | **No** (`.adv-only` tab) | — | Entire tab button `<button class="tab adv-only" data-tab="serve">` |
| **System** (`panel-system`) | **No** (`.adv-only` tab) | — | Entire tab button `<button class="tab adv-only" data-tab="system">` |
| **Capable** (`panel-capable`) | **Yes** | Full hardware capability assessment card (`#capVerdict`, `#capSlmRows`, `#capGgufRows`, `#capLimits`) — helps beginners see what their CPU/RAM can run. | None (read-only informational tab). |

---

## Implementation Tasks

### Task 1: Mode Switch Module (`ui/core/mode.js`), Header Toggle & Collapsible Log Dock

**Files:**
- Create: `ferrum_gui/ui/core/mode.js`, `ferrum_gui/ui/tests/mode.test.js`
- Modify: `ferrum_gui/ui/index.html` (add `#uiMode` segmented button / select in `<header class="topbar">`, add `#btnToggleLog` in `#terminal .term-bar`, tag `.adv-only` tabs)
- Modify: `ferrum_gui/ui/styles.css` (rules for `body[data-mode="beginner"] .adv-only`, step badges on Beginner tabs, collapsed `#terminal.collapsed` layout)
- Modify: `ferrum_gui/ui/core/app.js` (wire `initMode`), `ferrum_gui/ui/core/help.js` (add tooltips for `#uiMode` and `#btnToggleLog`)
- Modify: `ferrum_gui/ui/tests/modules.test.js`, `ferrum_gui/ui/tests/testapp.js`, `docs/gui-tooltips.md`

**Interfaces (`ui/core/mode.js`):**
```javascript
export const MODE_KEY = "ferrum.ui.mode.v1";

export function getMode() {
  return document.body?.dataset?.mode === "beginner" ? "beginner" : "advanced";
}

export function setMode(mode, { persist = true } = {}) {
  const next = mode === "beginner" ? "beginner" : "advanced";
  document.body.dataset.mode = next;
  // Update #uiMode control, collapse/expand #terminal, and if active tab is .adv-only in beginner mode, activate "datasets"
}

export function initMode() { ... }
```
*Note on default mode for existing tests:* When `localStorage.getItem(MODE_KEY)` is unset, default to `"beginner"` in interactive browser sessions or `"advanced"` if needed by legacy flow tests, **or** since `.adv-only` only sets CSS `display: none` (which the headless stub DOM in `testdom.js` ignores when dispatching clicks), defaulting to `"beginner"` works seamlessly across all existing `app_flows.test.js` tests! Verify this explicitly in Step 1.

**Steps:**
- [ ] **Step 1: Write `ferrum_gui/ui/tests/mode.test.js`** (per rule 19, every new file gets unit tests in the same commit):
  - `mode_switches_body_dataset_and_persists_to_local_storage`
  - `switching_to_beginner_on_adv_only_tab_redirects_to_datasets`
  - `switching_to_beginner_collapses_terminal_and_toggle_reopens_it`
- [ ] **Step 2: Create `ferrum_gui/ui/core/mode.js`** implementing `getMode`, `setMode`, `toggleLogDock`, and `initMode`.
- [ ] **Step 3: Update `ferrum_gui/ui/index.html` and `ferrum_gui/ui/styles.css`:**
  - Add `<select id="uiMode" aria-label="Interface mode"><option value="beginner">Beginner mode</option><option value="advanced">Advanced mode</option></select>` in `.topbar`.
  - Add `<button type="button" id="btnToggleLog" class="btn-small">Hide / Show</button>` in `.term-bar`.
  - Add `class="tab adv-only"` to `prepare`, `models`, `finetune`, `export`, `serve`, `system` tab buttons.
  - Add CSS rules in `styles.css`:
    ```css
    body[data-mode="beginner"] .adv-only { display: none !important; }
    body.log-collapsed #app { grid-template-rows: 52px 1fr 32px; }
    body.log-collapsed #logOutput { display: none; }
    ```
- [ ] **Step 4: Add help entries for `uiMode` and `btnToggleLog` in `ferrum_gui/ui/core/help.js`** (`>= 30` words each) and document them in `docs/gui-tooltips.md`.
- [ ] **Step 5: Run all JS checks and tests:**
  - `cd ferrum_gui && for f in $(find ui -name '*.js'); do node --check "$f" || exit 1; done && for t in ui/tests/*.test.js; do node "$t" || exit 1; done`
  - `cd ferrum_gui && CARGO_BUILD_JOBS=3 cargo test`
- [ ] **Step 6: Commit:** `feat(gui): add Beginner/Advanced mode switch and collapsible engine log`.

---

### Task 2: Train Tab Model Size Presets (`#trPreset`) & Beginner Field Curation

**Files:**
- Modify: `ferrum_gui/ui/index.html` (add `#trPreset` select at top of `#panel-train`; tag advanced inputs/fieldsets across Datasets, Train, Generate, Evaluate, GGUF with `class="adv-only"`)
- Modify: `ferrum_gui/ui/tabs/train_tab.js` (export `PRESETS` matching `slm_cli/src/preset.rs`; wire `#trPreset` `change` listener to populate the 7 fields and trigger `schedulePreflight()`; wire manual edits on the 7 fields to set `#trPreset` to `"custom"`)
- Modify: `ferrum_gui/ui/core/help.js` (add `trPreset` tooltip entry with `>= 30` words and live estimate support)
- Modify: `ferrum_gui/ui/tests/app_flows.test.js`, `ferrum_gui/ui/tests/settings.test.js`, `ferrum_gui/ui/tests/a11y_html.test.js`, `docs/gui-tooltips.md`

**Preset Table (matching `slm_cli/src/preset.rs`):**
| Preset value | Label | `trContext` | `trEmbed` | `trHeads` | `trBlocks` | `trHidden` | `trVocab` | `trLr` |
|---|---|---|---|---|---|---|---|---|
| `custom` | `Custom (manual settings)` | *(unchanged)* | *(unchanged)* | *(unchanged)* | *(unchanged)* | *(unchanged)* | *(unchanged)* | *(unchanged)* |
| `tiny` | `Tiny (~0.2M params — minutes on CPU)` | `64` | `64` | `4` | `2` | `256` | `512` | `0.01` |
| `small` | `Small (~1.5M params — hours on CPU)` | `128` | `128` | `4` | `4` | `512` | `2048` | `0.003` |
| `medium` | `Medium (~7M params — 1–2 days on CPU)` | `256` | `256` | `8` | `6` | `1024` | `4096` | `0.001` |

**Steps:**
- [ ] **Step 1: Write unit/flow tests** in `ferrum_gui/ui/tests/mode.test.js` and `app_flows.test.js`:
  - `selecting_train_preset_populates_architecture_fields_and_updates_preflight`
  - `editing_embed_dim_switches_preset_select_to_custom`
  - `every_panel_has_minimal_visible_controls_in_beginner_mode` (verifies the exact `.adv-only` tagging on `#panel-datasets`, `#panel-train`, `#panel-generate`, `#panel-evaluate`, `#panel-gguf`).
- [ ] **Step 2: Add `#trPreset` and `.adv-only` classes in `ferrum_gui/ui/index.html`** according to the Per-Tab Field Curation Matrix above.
- [ ] **Step 3: Wire `#trPreset` in `ferrum_gui/ui/tabs/train_tab.js`** and add `trPreset` to `ferrum_gui/ui/core/help.js` and `docs/gui-tooltips.md`.
- [ ] **Step 4: Refresh fixtures if needed and run all JS + Rust tests.**
- [ ] **Step 5: Commit:** `feat(gui): add model size presets on Train tab and curate Beginner mode forms`.

---

### Task 3: Guided "Next Step" Workflow Banners (Datasets → Train → Generate / Evaluate)

**Files:**
- Modify: `ferrum_gui/ui/index.html`:
  - In `#panel-datasets`: add `<div id="dsNextStep" class="next-step-banner" hidden><span id="dsNextText">Corpus saved.</span> <button type="button" id="btnGoTrain" class="primary">Next: Train a model on this corpus →</button></div>`
  - In `#panel-train`: add `<div id="trNextStep" class="next-step-banner" hidden><span id="trNextText">Model trained!</span> <button type="button" id="btnGoGenerate" class="primary">Next: Generate text →</button> <button type="button" id="btnGoEvaluate">Evaluate model →</button></div>`
- Modify: `ferrum_gui/ui/tabs/datasets_tab.js` (unhide `#dsNextStep` when `save_corpus` succeeds; clicking `#btnGoTrain` sets `#trCorpus.value = outPath` and activates `[data-tab="train"]`)
- Modify: `ferrum_gui/ui/tabs/train_tab.js` (unhide `#trNextStep` when `train_slm` completes without error; clicking `#btnGoGenerate` sets `#genModel.value = outPath` and activates `[data-tab="generate"]`; clicking `#btnGoEvaluate` sets `#evModel.value = outPath` and activates `[data-tab="evaluate"]`)
- Modify: `ferrum_gui/ui/core/help.js`, `ferrum_gui/ui/styles.css`, `docs/gui-tooltips.md`
- Modify: `ferrum_gui/ui/tests/app_flows.test.js`

**Steps:**
- [ ] **Step 1: Write flow tests** in `ferrum_gui/ui/tests/app_flows.test.js`:
  - `saving_corpus_reveals_next_step_button_that_switches_to_train_with_corpus_filled`
  - `finishing_train_reveals_next_step_buttons_for_generate_and_evaluate`
- [ ] **Step 2: Add the banner HTML in `index.html`, styling in `styles.css`, and event handlers in `datasets_tab.js` and `train_tab.js`.**
- [ ] **Step 3: Add help entries for `btnGoTrain`, `btnGoGenerate`, and `btnGoEvaluate` in `help.js`** (`>= 30` words each) and update `docs/gui-tooltips.md`.
- [ ] **Step 4: Run all Node and Rust tests.**
- [ ] **Step 5: Commit:** `feat(gui): add guided Next Step workflow buttons across Datasets, Train, Generate, and Evaluate`.

---

### Task 4: Mode-Aware Tooltips (`ui/core/tooltip.js`)

**Problem:** Currently `renderCard` in `ferrum_gui/ui/core/tooltip.js` (lines 50–77) always renders all four sections (`What it is`, `How it works`, `New to AI?`, `With your current settings`), producing an 80–120 word popover on every hover.

**Files:**
- Modify: `ferrum_gui/ui/core/tooltip.js`
- Modify: `ferrum_gui/ui/tests/help.test.js`, `ferrum_gui/ui/tests/mode.test.js`
- Modify: `docs/gui-tooltips.md`, `Manual.md` §10.3, `Status.md` §5 & §6.2

**Interfaces (`renderCard` in `ferrum_gui/ui/core/tooltip.js`):**
- When `document.body?.dataset?.mode === "beginner"`:
  1. **New to AI?** (`entry.beginner`) — placed first so plain-English guidance is at the top.
  2. **What it is** (`entry.what`)
  3. **With your current settings** (`liveText`, if available)
  *(Omits `How it works` in Beginner mode, with a subtle footer hint: `Switch to Advanced mode for technical mechanics.`)*
- When `document.body?.dataset?.mode !== "beginner"` (Advanced mode or unset):
  1. **What it is** (`entry.what`)
  2. **How it works** (`entry.how`)
  3. **With your current settings** (`liveText`, if available)
  4. **New to AI?** (`entry.beginner`) — kept in Advanced mode if `help.test.js` asserts all 3 static sections in default DOM, **or** rendered as a compact secondary note so existing `help.test.js` assertions (`What it is`, `How it works`, `New to AI?`) pass in default test mode while Beginner mode streamlines the card.

**Steps:**
- [ ] **Step 1: Write unit tests** in `ferrum_gui/ui/tests/mode.test.js`:
  - `tooltip_in_beginner_mode_leads_with_new_to_ai_and_omits_how_it_works`
  - `tooltip_in_advanced_mode_includes_technical_how_it_works`
- [ ] **Step 2: Update `renderCard` in `ferrum_gui/ui/core/tooltip.js`.**
- [ ] **Step 3: Run all Node tests (`help.test.js`, `tooltips_guard.test.js`, `mode.test.js`, `app_flows.test.js`) and `cargo test` in `ferrum_gui/`.**
- [ ] **Step 4: Update `docs/gui-tooltips.md`, `Manual.md` §10.3, and `Status.md` §5/§6.2.** Commit: `feat(gui): tailor tooltip sections to Beginner and Advanced mode`.

