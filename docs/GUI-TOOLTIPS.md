<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Tutorial: tool-tips in Ferrum SLM Studio

Every control and every informative element of the Studio window has a
tool-tip of at least 30 words: the tabs, buttons, fields, checkboxes and
selects, the section headings, box titles and introduction paragraphs, the
status lines, progress bars, error lines, table headers, charts, result and
output areas, the top bar figures and the two dialogs. This page explains how they work, how to add
one, and the test that keeps the rule.

## 1. Using them

- **Mouse.** Rest the pointer on an element. The tip appears after 350 ms
  (`HOVER_MS` in `ferrum_gui/ui/core/tooltip.js`) and goes when the pointer leaves.
  Hovering the text label of a field shows the field's tip.
- **Keyboard.** Tab to a control and its tip shows at once. It hides on blur.
  The two logs (`#termOut`, `#svLog`) can also be focused, so they show their tip
  too.
- **Escape** hides the open tip. It stays hidden until the pointer or focus
  leaves that element and comes back. Scrolling and resizing the window also
  hide it.
- **Screen readers.** Every tipped element carries `aria-describedby`, which
  points at a visually hidden copy of the text. On a control, a screen reader
  announces it when focus reaches the control, whether or not the bubble is
  shown.
- **Hover only for the rest.** Headings, hint paragraphs, status lines and other
  elements that are not controls are **not** in the Tab order, so the keyboard
  cannot open their bubble. Whether a screen reader reads their
  `aria-describedby` text while browsing depends on the screen reader. More
  than a hundred extra Tab stops would make keyboard use of the forms slower.
- **The hardware line** (`#hwLine`, marked `data-tip-live`) adds its current
  text, which the top bar may cut short, to its tip under *Now*.

The bubble is one shared `<div role="tooltip" id="ferrum-tooltip">`. It is
styled by `.tip` in `styles.css`, using the theme variables (`--bg3`, `--fg`,
`--accent`) and a width of at most 440 px. `placeTooltip` puts it below the
element, or above when there is more room there. It is always clamped inside the
window. Inside an open modal dialog the bubble moves into the dialog, so it
cannot render behind it. Text is set with `textContent` only; the CSP forbids
inline styles, and positioning uses CSSOM (`element.style.left`).

## 2. Where a tip's text comes from

There are two sources, and an element uses exactly one.

1. **Controls** (`input`, `select`, `textarea`, `button` with an id) take their
   text from their entry in `ui/core/help.js`. The card follows the UI mode
   (`body[data-mode]`, set by `ui/core/mode.js`). **Beginner:** *New to AI?* then
   *What it is* (field) or *What it does* (button), then the live estimate; *How it
   works* is left out (kept only when an entry has no `New to AI?` text). **Expert**
   (or no mode set): *What it is / What it does*, then *How it works*, then the live
   estimate; *New to AI?* is left out. The 30-word guard counts an entry's whole text,
   since each mode shows a subset. The entries added with the mode feature are
   `uiMode` (interface mode switch), `btnToggleLog` (engine log show/hide), `trPreset`
   (model size preset), `trMaxSeconds` (Train time budget), `trMetricsCsv` (Train per-step CSV log), and `btnGoTrain`, `btnGoGenerate`, `btnGoEvaluate` (the
   guided next-step buttons). A button also gets the live time
   and RAM estimate that `live.js` adds. These entries existed before this
   feature, and 31 of them (the 12 tabs and 19 buttons) were extended to reach
   30 words; `help.test.js` checks them.
2. **Everything else** has a `data-tip` attribute in `ui/index.html`, holding
   plain text with no markup, `&`, `<`, `>` or `"`. `live.js` attaches it with
   `tooltip.contentFromTip`, which gives one section and no title. With
   `data-tip-live` the element's current text is appended as a second section.

Controls created at runtime use the same two paths:

- The catalog's Download buttons set `data-help="dsCatalogDownload"`.
- The *Show full message* and *Copy details* buttons of an error line set a
  `data-tip` (`ERR_TIPS` in `common.js`).
- In both cases the module then calls `live.attach(button)` through `getLive()`.

Content rendered into a container is covered by the container's tip. That
covers dataset cards, Evaluate rows, cards, per-core bars and log lines.

## 3. Adding a tip

To give a new element a tip, follow these steps.

1. Write what the element does or shows: units, defaults and ranges, what a
   click does and any side effects, and when it is hidden or disabled. Take
   every fact from the code (the tab module in `ui/tabs/*_tab.js` and the Rust
   command in `ferrum_gui/src/`). Do not take facts from memory. Do not pad the
   text by repeating the label.
2. Make it at least 30 words. The count is whitespace-separated pieces, as
   `tooltip.wordCount(text)` counts them.
3. Put the text in the right place:
   - **A new control:** add a `help.js` entry under the control's id. Use
     `does` plus `how` for a button, and `what`, `how` and `beginner` for a
     field. Do not add a `data-tip` to a control.
   - **Any other element:** add `data-tip="..."` to its tag in `index.html`.
   - **A control built at runtime:** set `data-help` (for a help entry) or
     `data-tip` on it, then call `live.attach(el)`.
4. Do not use the native `title` attribute. It would open a second bubble, and
   the guard test refuses it.
5. Run the guard:

```bash
cd ferrum_gui
node ui/tests/tooltips_guard.test.js
```

## 4. The guard test

`ui/tests/tooltips_guard.test.js` (`every_element_has_a_30_word_tooltip`) parses
`index.html` with the dependency-free reader in `ui/tests/testhtml.js`. It fails with
a list of every element that needs a tip and has fewer than 30 words. An
element needs its own tip when it is a control, an `img`, `canvas`, `pre`,
`table`, heading, `legend` or `th`, or has an id, a role, an `aria-label` or
text of its own. These elements are exempt, and the reasons are listed in the
test:

| Exempt | Why |
|---|---|
| `<option>` | the `<select>`'s tip covers its options |
| `aria-hidden="true"` (the spinners) | decorative |
| `b`, `strong`, `em`, `i`, `code`, `br` | inline text; the tip of the text block around it covers it |
| `#topbar`, `#miniStats`, `#tabs`, `#content`, `#svRunning`, `#terminal`, `#capDialog`, `#ggWarnDialog`, every `role="tabpanel"` | page background and wrappers; each of their parts has its own tip |
| anything inside a `role="progressbar"` | the bar's tip covers its fill |
| a `<label>` wrapping a control with a 30-word tip | the label shows that control's tip on hover |
| any other element with no id, role, `aria-label` or text of its own (layout `div`, `span`, `section`, `fieldset`, `form`, `menu`, `thead`, `tbody`, `tr`) | structural layout |

The test also checks five more things:

- 29 words fail and 30 pass, on synthetic markup.
- No control has both a help entry and a `data-tip`.
- No `data-tip` contains markup characters.
- No element uses `title`.
- The controls built at runtime have tips of at least 30 words: the catalog's
  Download buttons (help entry `dsCatalogDownload`) and the two error-line
  buttons (`ERR_TIPS` in `common.js`). No other tab module creates a control.
- The real page passes. On 2026-11-03 that was 352 elements with their own
  tip: 227 controls and 124 `data-tip` elements, 12 of them the hint
  paragraphs.

Before the tips were written, the guard listed 143 offenders on the page:

- 31 controls whose help text was under 30 words, among them all 12 tabs.
- 112 elements with no tip at all.

The Serve tab's Expert-mode controls `svQuant`, `svRepPenalty` and `svForce`
(2026-11-07, H4) have help entries in `help.js` (each over 30 words, the same
rules as above); `svForce` ends in `Force`, so settings never remember it.

## 5. Tests that cover the component

- `ui/tests/tooltip.test.js` covers:
  - placement, as a pure function;
  - hover delay with a fake timer;
  - focus showing the tip at once;
  - Escape;
  - live refresh;
  - no markup injection;
  - dialogs;
  - `wordCount`, `contentFromTip` (with and without live text) and the
    title-less bubble.
- `ui/tests/app_flows.test.js` runs the real `app.js` on the real page (stub DOM). It
  checks four flows:
  - hovering a tab shows its help tip;
  - hovering a heading shows its `data-tip` as written, with the same text for
    screen readers;
  - focusing the engine log shows its tip and Escape hides it;
  - the runtime error buttons get their tips.

None of this was click-tested in a real window. The layout of the bubble on the
real screen has not been seen.
