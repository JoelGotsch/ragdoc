# Drawing diagrams

Every diagram is hand-drawn inline SVG inside a `<figure class="diagram">` (see
`components.md`). Pick the type by the question the reader has:

| question | type | `data-type` |
|---|---|---|
| who calls whom, in what order, for one request | sequence | `sequence` |
| which branches exist and where each one ends | flowchart | `flow` |
| how the call path changed between base and head | call-stack diff | `call-stack-diff` |

Draw at most two diagrams for a review (one in Design, one in Implementation) unless a third
answers a question the first two cannot.

## Skeleton

```html
<svg viewBox="0 0 1120 620" style="min-width:1120px" role="img"
     aria-labelledby="SLUG-title SLUG-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="SLUG-title">One line naming what the diagram shows.</title>
  <desc id="SLUG-desc">Two or three sentences a screen-reader user can follow instead.</desc>
  <defs>
    <marker id="SLUG-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#4f5d75"/>
    </marker>
    <marker id="SLUG-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#bf4520"/>
    </marker>
  </defs>
  <style>
    .SLUG-name { font: 600 11px 'Geist', system-ui, sans-serif; fill: #2d3142; }
    .SLUG-sub  { font: 500 9px 'Geist Mono', ui-monospace, monospace; fill: #5a6580; }
    .SLUG-msg  { font: 500 10px 'Geist', system-ui, sans-serif; fill: #2d3142; }
    .SLUG-eyebrow { font: 500 7px 'Geist Mono', ui-monospace, monospace; fill: #5a6580; letter-spacing: 0.18em; text-transform: uppercase; }
    .SLUG-note { font: italic 14px 'Instrument Serif', serif; fill: #4f5d75; }
  </style>
  <!-- 1 connectors, 2 boxes, 3 labels, 4 annotations -->
</svg>
```

Every `id` and class inside the SVG starts with the slug: the page may hold several diagrams
and the exporter namespaces `<defs>` by slug. The `<style>` is scoped by those class names and
is carried into the exported file.

## Sequence

- Lifelines: ≤5. Actor box 160×40 at the top (`rx=6`, backend treatment), centred on the
  lifeline x; lifelines 220px apart on a 1120 canvas (x = 120, 340, 560, 780, 1000). Lifeline:
  `<line x1=x y1=64 x2=x y2=H-24 stroke="#2d3142" stroke-opacity="0.55" stroke-dasharray="3,3"/>`.
- Activation bar: `<rect x=x-4 width=8 fill="#2d3142" fill-opacity="0.06" stroke="#4f5d75" stroke-width="0.8"/>`
  spanning the messages it serves.
- Messages: ≤12, 36–50px apart, as `<g class="msg" data-k="msg" data-source="…">` holding one
  horizontal `<line>` with a marker and one `.SLUG-msg` label 6px above the line, left-aligned
  4px right of the start lifeline (or right-aligned for leftward messages). Calls are solid;
  returns are dashed `5,4`. A self-message is `M x,y H x+40 V y+20 H x+8` with the marker on the
  end and the label to the right of the loop.
- One fragment at most: a dashed `rx=8` rect with an eyebrow tag in its top-left corner
  (`loop · per changed source`) on its own paper mask, since lifelines cross it. Put it behind
  the messages it encloses. Every label that a lifeline or connector crosses needs a paper mask.
- Accent: the one or two messages that carry the behavioural change.

## Flowchart

- Nodes: ≤9. Start/end `rx=20` ovals 40px tall; steps `rx=6` rects 48px tall; decisions as
  `<polygon>` diamonds (half-width 90, half-height 32) with ≤3 exits, every exit labelled with an
  8px mono label (`yes`, `no`, `timeout`).
- Edges: ≤12, orthogonal, markers on the end, accent on the happy path only.
- Keep the happy path in one column and push failure branches to the right.
- Group as `<g class="node" data-k="node">` and `<g class="edge" data-k="edge">`.

## Annotation callout

An italic serif note (14px, ≤3 `<tspan>` lines, 260px wide) tied to the element it explains by a
dashed `4,3` Bézier leader ending in a 2px dot: `M nx,ny C …  tx,ty` plus
`<circle cx=tx cy=ty r=2 fill="#4f5d75"/>`. At most two per diagram; they are for the one fact
the geometry cannot show (a guarantee, a race, a limit).

## Order of drawing, then the gate

1. Connectors (so boxes mask them). 2. Boxes with their paper mask. 3. Labels. 4. Annotations.
Then `check_review.py`: budgets, ids, `aria`, grid, min-width, diagonals. Then `export_diagrams.py`
and open the `.svg` once to confirm fonts and markers survived the export.
