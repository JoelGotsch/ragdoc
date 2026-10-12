# Style guide

The visual system is adopted from Cathryn Lavery's
[diagram-design](https://github.com/cathrynlavery/diagram-design) (MIT; see
`../LICENSE-diagram-design.md`), trimmed to what a review page needs. The page CSS in
`assets/template.html` defines the tokens; diagrams repeat them as literal values because an
exported SVG has no page CSS.

## Tokens (light only)

| token | value | use |
|---|---|---|
| paper | `#f5f5f5` | page background, label masks |
| paper-2 | `#ececec` | secondary fill |
| ink | `#2d3142` | text, default strokes |
| ink-strong | `#111111` | diagram titles |
| muted | `#4f5d75` | secondary text, arrow strokes and markers |
| soft | `#5a6580` | sublabels, eyebrows |
| rule | `rgba(45,49,66,0.12)` | hairlines, borders |
| rule-solid | `#bfc0c0` | borders and badge fills only, never text |
| faint | `#666e80` | quietest text allowed (line numbers): ≥ 4.6 : 1 on white and paper |
| accent | `#bf4520` | ≤2 focal elements per diagram, warnings |
| accent-tint | `rgba(191,69,32,0.08)` | focal fill |
| link | `#2e5aa8` | hyperlinks, "success" callout |

## Contrast

Every text colour must reach 4.5 : 1 against both white and paper, and every stroke that carries
meaning (lifelines, fragment and removed-frame borders, connectors) 3 : 1, which for ink means a
`stroke-opacity` of at least 0.55. Hairline `rule` borders are decoration and exempt.
`check_review.py` enforces both. The page is light only: it declares `color-scheme: only light`
and `<meta name="darkreader-lock">` so browser auto-dark modes do not lighten the text while the
white panels and literal SVG fills stay bright. Do not remove them.

## Type

- Instrument Serif: page `h1`/`h2`, italic notes and annotation callouts (14px in SVG).
- Geist 600 11–12px: node and frame names. Geist 400/500 10px: sequence messages.
- Geist Mono: 9px sublabels (`path:L10-L24`), 7–8px eyebrows (`letter-spacing: 0.18em`,
  uppercase), 8px arrow labels (`letter-spacing: 0.06em`).

Font link (the only remote resource a page may load):
`https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap`

## Geometry

- 4px grid for every coordinate and the viewBox. Node widths 80–320, gaps 20/24/32/40/48,
  corner radius 4 (tags), 6 (boxes), 8 (elbows, panels). Strokes 0.8 / 1 / 1.2.
- A diagram is 960 or 1120 wide; the SVG carries `style="min-width:<width>px"` and the figure's
  `.diagram-container` scrolls horizontally on narrow screens.

## Node treatments (fill / stroke)

| treatment | fill | stroke | when |
|---|---|---|---|
| focal | accent-tint | accent 1.2 | the ≤2 things the reader must look at |
| backend | `#fff` | ink 1 | ordinary code, new or changed |
| store | ink @ 0.05 | muted 1 | stores, queues, files |
| external | ink @ 0.03 | ink @ 0.55 | third-party services |
| input | muted @ 0.10 | soft 1 | caller, request |
| optional / removed | ink @ 0.02 | ink @ 0.55 dashed `4,3` | gone on head, optional paths |

Every node is a paper-coloured mask rect under a styled rect, drawn after the arrows.

## Connectors

Orthogonal only: `M x1,y1 H mid-8 Q mid,y1 mid,y1+8 V y2-8 Q mid,y2 mid+8,y2 H x2`. Markers
`arrow` (muted), `arrow-accent`, `arrow-link`: `markerWidth 8 markerHeight 6 refX 7 refY 3`
polygon `0 0, 8 3, 0 6`. Returns are dashed `5,4`; lifelines dashed `3,3` at ink @ 0.55. Labels sit
above a horizontal segment or on a paper mask with 6–10px of gap; crossing strokes hop with
`a 8,8 0 0,1 16,0`; fan attach points are ≥12px apart; no connector passes behind a box that is
not its endpoint.

## Budgets

| diagram | limit |
|---|---|
| sequence | 5 lifelines, 12 messages, 1 fragment |
| flow | 9 nodes, 12 edges, every branch labelled, ≤3 exits per decision |
| call-stack-diff | 10 frames per side |
| any | 2 accents, 2 annotation callouts |

Over budget means two diagrams (overview + detail), not a denser one.

## Page

Prose width 78ch. Sections in fixed order (what/why, design, implementation, files) with a
`nav.toc`. Callouts and peeks live inside the prose column; diagrams span the full frame. No
JavaScript, no external CSS, no images other than inline SVG.
