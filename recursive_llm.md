# Recursive VLM Calls Approach

## Overview

A positioning strategy that relies entirely on a vision LLM (VLM) to locate extracted data
on the page, with no OCR step at all. Every item that needs a precise position is located by
directly asking the VLM to point to it within a cropped region of the drawing, recursively
subdividing that region whenever too many items need to be located within it for one call to
handle reliably.

This is the same recursive-splitting mechanism used as the fallback stage inside the hybrid
OCR + VLM approach, described here as its own standalone strategy: what it would look like
if it were used for every item from the start, rather than only for OCR's leftovers.

## How It Works

### Region Splitting Logic

Start with a region of the page (e.g., a grid cell, or a whole named view) and the list of
items known to be located somewhere inside it. If that list is short enough, crop the
region, upscale it so text is large and legible, and ask the VLM to point to each item in
one call. If the list is too long for one reliable call, split the region in half along its
longer axis (with a small overlap so nothing sitting on the split line is missed by both
halves), and recurse into each half with the same item list, minus whatever was already
found in the first half. This repeats, region getting smaller and less cluttered each time,
until either every item is found or a maximum recursion depth is reached.

### Identity Preservation

Every item carries its own stable unique ID, assigned once during the initial semantic
extraction and never regenerated. The VLM itself never needs to see or transcribe that ID
directly; each individual call uses small local reference numbers for the model to point at,
which are translated back to the real stable ID in code before merging results. This keeps
every recovered position correctly tied to its original row, no matter how many times the
region has been split or how deep the recursion goes.

## Cost and Performance Characteristics

**This approach is mostly accurate, but it takes significantly more time and costs
significantly more than OCR-based or hybrid methods.**

- Every single item requires at least one VLM call chain to locate, rather than being
  resolved for free by OCR first. On a dense drawing, a region may need several rounds of
  splitting before every item is found, meaning multiple VLM calls just to fully resolve one
  small area of the page.
- Total call count, latency, and API cost scale directly with how much data is on the page
  and how densely packed it is, with no cheap first pass to absorb the easy majority of
  cases. A page with many items packed into a small area can require a meaningfully larger
  number of calls than a sparse one.
- Because it is fundamentally a visual pointing task performed by the VLM rather than
  character recognition, it tends to succeed more consistently than OCR on small, dense, or
  rotated text, which is where OCR-based methods most often fail. This is the source of its
  higher accuracy.
- Runtime is bound by network latency and per-call processing time on the gateway, repeated
  for every recursive split, which compounds on genuinely dense drawings.

## Strengths

- More consistently accurate than OCR on the specific cases OCR struggles with: small,
  dense, cluttered, or rotated dimension text
- Does not depend on OCR being installed, configured, or working correctly at all
- Scales its own effort automatically to match how visually dense a given region is,
  without needing per-drawing tuning

## Limitations

- Meaningfully slower end-to-end than any approach that uses OCR as a first pass, since
  every item costs at least one VLM round-trip rather than being resolved for free
- Meaningfully more expensive in API/compute cost for the same reason, especially on dense
  drawings that require several rounds of recursive splitting
- Not proportionally worth it for drawings that are mostly sparse, since the same cost is
  paid even where OCR alone would have succeeded easily

## When to Use

Best reserved for cases where OCR-based positioning has already been tried and found
insufficient (e.g., unusually dense or heavily rotated-text drawings), or for a smaller
batch of high-priority drawings where positioning accuracy matters more than turnaround
time or cost. Not recommended as the default, first-choice method for routine, high-volume
extraction runs.
