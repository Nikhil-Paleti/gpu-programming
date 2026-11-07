# Kogge–Stone vs Brent–Kung


## Kogge–Stone Scan

### Core idea
Kogge–Stone computes prefix sums by **repeated doubling**.

At each step, every element pulls data from a fixed distance behind it.

> **After step `k`, element `i` contains the sum of the previous `2^k` elements ending at `i`.**

### What happens at each step
- Step 0 (stride = 1): sum last 2 elements
- Step 1 (stride = 2): sum last 4 elements
- Step 2 (stride = 4): sum last 8 elements
- …
- After `log2(n)` steps: full prefix sum

Every round:
- Most threads participate
- Every participating element performs an addition

### Properties
- **Depth:** `log2(n)`
- **Work:** `O(n log n)` additions
- **Pros:** simple, low latency
- **Cons:** high shared-memory traffic, more total work

### When it’s good
- Warp-level scans (using shuffles)
- Small `n`
- When latency matters more than total work

---

## Brent–Kung Scan (Work-Efficient Scan)

Brent–Kung is best understood as **two separate phases**, each with its own invariant.

---

## Phase 1: Upsweep (Reduction)

### What is happening conceptually

This phase is **not computing prefix sums** yet.
It is building a **reduction tree**.

> **After stride `s`, the rightmost element of every `2s` segment stores the sum of that entire segment.**

- At stride 1 → every 2-element segment has its sum at the right boundary
- At stride 2 → every 4-element segment has its sum at the right boundary
- At stride 4 → every 8-element segment has its sum at the right boundary
- …
- At the end → the last element contains the total sum of the block

This phase is simply a **parallel reduction**.

---

## Phase 2: Downsweep (Prefix Propagation)

### What is happening conceptually

We already know the sum of every large segment.
Now we **propagate prefix information down the tree**.


> **At stride `s`, the prefix sum of a parent segment is added to its right child segment.**

Or equivalently:

> **Prefix sums are routed from parent nodes to right subtrees in the reduction tree.**

This fills in all missing prefix values.

---

## Combined Brent–Kung Mental Model

> *“Brent–Kung works in two phases. In the upsweep, we build a reduction tree so that the rightmost element of each power-of-two segment holds that segment’s total sum. In the downsweep, we traverse the tree in reverse and propagate prefix sums from parent segments to their right children, which fills in all prefix values. This reduces total work compared to Kogge–Stone at the cost of additional synchronization steps.”*

---

## Side-by-Side Invariant Comparison

### Kogge–Stone

> After step `k`, element `i` contains the sum of the previous `2^k` elements ending at `i`.

- Every element updates every round
- Simple mental model

### Brent–Kung

> Upsweep: right boundary of each `2^k` segment holds the segment sum.
>
> Downsweep: prefix sums are propagated from parent segments to right children.

- Only segment boundaries update
- Work-efficient

---

## Work vs Depth

| Algorithm      | Depth              | Total Work        |
|---------------|--------------------|-------------------|
| Kogge–Stone   | `log2(n)`          | `O(n log n)`      |
| Brent–Kung    | `~2 log2(n)`       | `~2n` (linear)    |

### Interpretation
- **Kogge–Stone:** fewer rounds, more total additions
- **Brent–Kung:** more rounds, fewer total additions

---

## GPU Performance Intuition

### Kogge–Stone
- Fewer `__syncthreads()`
- Heavy shared-memory traffic
- Good for warp-level scans

### Brent–Kung
- More synchronization points
- Much less shared-memory traffic
- Better for large block scans

In practice, **modern GPU libraries use hybrids**:
- Warp-level scan with shuffles (KS-like)
- Scan warp totals
- Add offsets

---

## Bites (Memorize These)

- “Kogge–Stone is latency-efficient but work-inefficient.”
- “Brent–Kung is work-efficient but has higher depth.”
- “Upsweep is a reduction; downsweep propagates prefixes.”
- “Production scans are hierarchical, not textbook.”

---
