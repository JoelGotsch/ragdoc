# Call-stack diff

The one diagram type that does not exist in diagram-design. It answers *how did the call path
change*: two columns of stack frames, base on the left and head on the right, rooted at the
same entry point, read top to bottom in call order with indentation for depth.

## When

Use it in the Implementation section when the change moved logic between functions or files
(an extraction, a split, a new engine under an old entry point). Do not use it for a change that
only edits bodies in place; a sequence diagram or a peek says that better.

## Content rules

- One entry point, the same on both sides (`VectorStorePipeline.run`). Follow one concrete
  request (one updated source, one failing batch), not every branch.
- ≤10 frames per side. Collapse siblings you do not discuss into one frame
  (`parse → process → split → chunk`). Depth ≤5.
- Every frame carries `data-source="base|head/path#L1-L2"` on its `<g>`; `resolve.py` verifies
  the range exists, and `check_review.py` counts frames by `data-k="frame"`.
- Badges, in the frame's top-right corner, are letters, not colours: `+` new on head, `−` gone
  on head, `Δ` same name but different body. Unbadged frames are unchanged pass-throughs.
- Accent (focal treatment) on ≤2 frames in total: where the guarantee changed.
- Removed frames (`−`) use the dashed "removed" treatment; everything else backend (`#fff`).

## Geometry (viewBox 1120 × H, H on the 4px grid)

| item | value |
|---|---|
| columns | base x 24–544, head x 576–1096 (gutter 32) |
| column header | eyebrow at y=16: `BASE · main @ 33cfbca5`, `HEAD · branch @ d74b653a` |
| frame | 400 × 40, `rx=6`; first at y=48, pitch 52; indent 24 per depth |
| frame text | name `.SLUG-name` at (x+12, y+17); `path:L1-L2` `.SLUG-sub` at (x+12, y+31) |
| badge | rect (x+372, y+13) 16×14 `rx=2` fill `#bfc0c0` @0.35; letter 9px mono centred |
| connector | from the parent's `(x+12, y+40)` down to the child's `(x+12, cy+20)` then right to the child's left edge: `M px+12,py+40 V cy+20 H cx`, muted 1px, `arrow` marker |
| legend | strip at the bottom (H − 36): `+ new on head`, `− gone on head`, `Δ changed body`, one focal sample |

Connectors are drawn before frames. A parent with several children fans from the same vertical
stem; the stem at `x+12` runs left of the children's boxes (which start at `x+24`), so it never
transits behind a non-endpoint box.

A frame whose work moved to another file on head is drawn on both sides with its own path on
each; the reader sees the move by comparing sublabels, which is the point of the diagram.
