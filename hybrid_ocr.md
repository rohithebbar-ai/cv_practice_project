# Hybrid OCR + VLM Approach

## Overview

A two-stage pipeline for locating extracted engineering drawing data on the page. Semantic
extraction (what something is, and its logical grouping) is always done once, by a vision
LLM (VLM), looking at the full drawing. Locating that data precisely on the page (where it
sits, in pixels) is then handled by two layered methods in sequence: a fast, free OCR pass
first, and a narrow, targeted VLM call only for whatever OCR could not confidently place.

The core idea: OCR is cheap and fast but struggles with small, dense, rotated, or cluttered
text. A VLM is much better at visually locating things in a busy image, but is slow and
costs money per call. Rather than choosing one or the other, this approach uses OCR to
handle the easy majority of cases for free, and reserves the expensive VLM call only for
the harder minority OCR could not resolve.

## Stage 1: Semantic Extraction (VLM, full page)

The VLM reads the entire drawing sheet in one pass and produces structured rows: what each
item is (Category: dimension / component / title block), its value, which named view or
detail it belongs to (View_Label), and which coarse grid cell it sits in (Dwg_View, e.g.
"G-11", read from the sheet's own printed row-letter/column-number border). This stage is
the only place semantic meaning and relationships are ever decided. Every row gets a stable
unique ID at this point, which is used to track it through every later stage.

## Stage 2: OCR-Based Positioning

The full page is OCR'd (at multiple rotations, since engineering drawings commonly print
height dimensions rotated 90 degrees along vertical dimension lines). For each extracted
row, its own text (component name or dimension value) is fuzzy-matched against OCR'd words,
searching only within that row's known grid cell (to avoid matching the wrong instance of a
repeated number elsewhere on the page). Where a confident match is found, the row gets a
real, precise pixel position for free.

## Stage 3: Targeted VLM Grounding (only for OCR's leftovers)

Rows OCR could not confidently place are grouped by grid cell. For each affected cell (not
each leftover row), the cell region is cropped out of the original image, upscaled so small
text becomes large and legible, and sent to the VLM with a short list of only the items
still missing from that cell, asking it to point to where each one sits within the crop. If
a cell has too many missing items for one call, the crop is split in half and the same
question is asked of each half, recursing until each crop is small enough to answer
reliably. This keeps the number of VLM calls proportional to how much OCR actually missed,
not proportional to total item count or page count.

## Relationship / Identity Preservation

Every row's stable ID (assigned once in Stage 1) is the only key ever used to write a
position back onto that row, at every stage. Positioning stages never re-interpret or
reconstruct meaning; they only ever answer "where," never "what." This guarantees a
recovered position can never be misattributed to the wrong row, and that Category,
Component_Name, and View_Label relationships established in Stage 1 are never altered by
anything that happens afterward.

## Strengths

- Fast and low-cost for the majority of cases, since OCR handles most items for free
- VLM cost and latency scale with how much OCR actually missed, not with total data volume
- Works even if the VLM/gateway is temporarily unavailable for the grounding stage (rows
  simply fall back to an approximate position instead of failing entirely)
- Identity/relationship-safe by construction (UUID-based merge at every stage)

## Limitations

- Positioning accuracy is capped by whatever OCR plus one targeted VLM pass can resolve;
  it does not guarantee 100% precise placement for every row
- Rows with no matching literal text on the page (e.g., a paraphrased description rather
  than an exact printed value) may never get a precise position from either stage
- Still depends on a reasonably accurate coarse grid system to constrain the search window
  and to know which cell to crop

## When to Use

This is the default, recommended approach for routine extraction runs where turnaround time
and cost matter and near-total (not literally 100%) positioning accuracy is acceptable, with
manual correction available for whatever remains approximate.
