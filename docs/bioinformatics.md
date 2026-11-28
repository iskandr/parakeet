# Bioinformatics & structural biology

Genomics, cheminformatics, and structural biology are full of kernels that do
not fit standard dense GEMM libraries: byte-level FASTQ parsing, 2-bit k-mer
packing, multi-channel pileups, 2048-bit fingerprint screening, 3D protein
message passing, and banded dynamic programming. Parakeet compiles all of them
directly from concise Python and NumPy into multi-core CPU and GPU code.

Every snippet below is included verbatim from
[`examples/bio_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_gallery.py).
Run the examples or benchmarks from a checkout:

```bash
python -m examples.bio_gallery
python benchmarks/bio_workloads.py
```

## Categorical encoding for sequences and tokens

Working with character sequences in compiled code—whether DNA bases (`ACGT`),
amino acids, GC indicator masks, or text tokens for a language model—comes down
to the same operation: mapping ASCII bytes to compact integer codes through a
256-entry lookup table.

Rather than defining an `IntEnum` and a separate dict mapping characters to
enum members, Parakeet provides two lightweight categorical encoders:

* **`par.Alphabet("ACGT")`** maps ASCII characters to `uint8` codes `0..3`
  (case-insensitive by default), and **`par.Alphabet.mask("GC")`** builds a
  0/1 indicator table. Both work outside `@par.jit` (`DNA.encode(reads)`) and
  inside `@par.jit` (`DNA[seq]`, `GC[seq]`), compiling to a single L1-resident
  256-byte gather.
* **`par.Vocab.from_text(text)`** builds the same 256-entry LUT from a text
  corpus for language-model tokenization (`vocab.encode(text)` /
  `vocab.decode(ids)`), as shown in [Train nanoGPT](gallery/nanogpt.md).

## How Parakeet compiles bio workloads

| Domain task | Operators & compiler passes | What is eliminated |
|---|---|---|
| FASTQ k-mers & read QC | Multi-output [`par.map`](operators.md#map) + `par.Alphabet` LUT | Manual `parfor`, scratch buffers, and ASCII bit-twiddling |
| Multi-channel BAM pileup | Keyed [`IndexScatter`](primitives.md#scatter) fusion | Redundant passes over alignment positions (`6 → 1`) |
| 2048-bit Tanimoto screening | `par.popcount` + [`par.allpairs`](operators.md#allpairs) + ordered [`par.fold`](operators.md#fold) | `O(N · M)` similarity matrix allocation |
| 3D protein RBF message passing | [Law 1 monoidal tiling](algebra.md#law-1-rescaled-semiring-monoids) + `par.grad` | `O(N²)` pairwise distance & weight matrices |
| Banded Smith-Waterman & DNA SSM | Two-row banded DP + [Law 4 chunkwise scan](algebra.md#law-4-chunkwise-state-space-duality) | Python DP interpreter overhead & serial scan bottlenecks |

---

## 1. FASTQ 2-bit canonical k-mers and read QC

Parakeet accepts `list[str]` directly (packing fixed-length strings into 2D
`uint8` or variable-length strings into `RaggedArray[uint8]`), compiles
`DNA[seq]` and `GC[seq]` to 256-byte LUT gathers, and unboxes multi-output
`par.map` tuples into one parallel pass:

```python
--8<-- "examples/bio_gallery.py:fastq_kmer_gc"
```

## 2. Single-pass multi-channel alignment pileup

All six `par.scatter_add` channels (A/C/G/T counts, Phred quality sum, reverse
strand count) share `ref_pos`, so scatter fusion merges them into one
thread-privatized pass (`IndexScatter: 6 → 1`):

```python
--8<-- "examples/bio_gallery.py:bam_pileup_fused"
```

## 3. 2048-bit Morgan fingerprint Tanimoto screening

Compares query fingerprints `(N, 32)` against a library `(M, 32)` of packed
`uint64` bitvectors using `par.popcount`, `par.allpairs`, and an ordered
`par.fold` (preserving first-index ties without allocating an `N × M` matrix):

```python
--8<-- "examples/bio_gallery.py:tanimoto_screen"
```

## 4. 3D protein coordinate RBF message passing

Pairwise radial-basis-function smoothing over 3D residue coordinates `(N, 3)`
uses the same [rescaled-monoid tiling](algebra.md#law-1-rescaled-semiring-monoids)
as softmax attention—zero `O(N²)` buffers and directly differentiable through
`par.grad`:

```python
--8<-- "examples/bio_gallery.py:protein_rbf"
```

## 5. Banded Smith-Waterman and DNA state-space scan

Two-row banded Smith-Waterman stays in `O(M)` L1 scratch, while 1D DNA
exponential recurrences rewrite automatically into
[Law 4 chunkwise parallel scans](algebra.md#law-4-chunkwise-state-space-duality):

```python
--8<-- "examples/bio_gallery.py:banded_sw_and_ssm"
```

---

## 6. Classic sequence & structural algorithms

[`examples/bio_algorithms.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_algorithms.py)
implements seven textbook algorithms in compiled Python/NumPy
([see head-to-head timings](release-benchmarks.md#bioinformatics-and-structural-biology-24-core-amd-epyc)):

| Algorithm | Function | Key Parakeet mechanisms |
|---|---|---|
| 3-state affine-gap Gotoh + traceback | `gotoh_global_align` | `M`/`Ix`/`Iy` DP tables, packed traceback, local growable CIGAR lists |
| Multi-word Myers bit-vector edit distance | `myers_edit_distance` | 64-bit `uint64` bit-parallel blocks (`Pv`, `Mv`, `Peq`) |
| `O(N)` monotone-deque minimizer sketch | `minimizer_sketch` | `par.Alphabet("ACGT", unknown=4)`, SplitMix64 hash, circular monotone deque |
| FM-index / BWT backward search | `build_fm_index`, `fm_backward_search_batch` | Sampled `Occ` checkpoints + parallel `RaggedArray[uint8]` batch queries |
| Parallel CIGAR expansion | `expand_cigar` | One-line `par.flatmap` (`IndexExpand`) |
| Profile HMM Viterbi + Forward-Backward | `hmm_viterbi_and_posterior` | Log-space decoding and posterior probabilities |
| Kabsch 3D superposition & RMSD | `kabsch_rmsd`, `batched_kabsch_rmsd` | Compiled `np.linalg.svd` + `det` (reflection-corrected) inside `par.parfor`, zero external LAPACK |

---

## 7. Long-read RNA alignment & quantification

Inspired by [Rob Patro](https://www.cs.umd.edu/~rob/)'s
[COMBINE lab](https://combine-lab.github.io/) tools (SSHash, Piscem, Spearfish,
and Oarfish), [`examples/patro_longread_rna.py`](https://github.com/iskandr/parakeet/blob/master/examples/patro_longread_rna.py)
implements a full long-read RNA aligner and isoform quantifier in ~250 lines of
compiled Python/NumPy:

1. **Unitig collapse + SSHash minimizer index (`build_unitig_sshash_index`):**
   collapses isoforms into maximal shared unitigs in CSR arrays with a
   `FrozenHashTable` mapping sparse minimizers to unitig buckets.
2. **Streaming unitig contiguity + Spearfish delta-mapping (`stream_map_long_reads`):**
   extends Maximal Exact Matches along unitigs in `O(1)`/base (skipping hash
   lookups), projects anchors to isoforms via CSR postings, chains co-linear
   diagonals, and keeps competitive isoforms within `delta_frac`.
3. **Oarfish-style coverage-aware EM (`oarfish_em_quantify`):** compiled EM
   over compatibility sets weighted by alignment score and effective length.

On 80 human Ensembl cDNA isoforms (`TP53`, `PKM`, `MAPT`, `ACTB`; 225,539 raw
bp), unitig collapse shrinks the reference **9.7×** (`23,247` bp across 231
unitigs), the SSHash fast path skips hash probes on **92.2%** of bases,
multicore mapping sustains **548.6 Mbp/s** on Apple M1 Pro, and Oarfish EM
reaches **Pearson r = 0.965** against ground truth.

---

## 8. Streaming I/O, BGZF inflate, and compiler scope

Parakeet intentionally keeps file descriptors, container framing, and streaming
I/O syscalls outside `@par.jit` so compiled kernels stay small, portable, and
GPU-safe. Because compiled functions **release the Python GIL**, disk/codec
work overlaps cleanly with compute:

* **Double-buffered I/O + compute:** a background reader thread decompresses
  slabs (`isal`, `zlib-ng`, `libdeflate`, or `pysam`) into pinned `uint8` /
  `RaggedArray` buffers while Parakeet processes the previous slab across CPU
  cores or on GPU (reserve 1–2 CPU cores for the reader on CPU-only hosts via
  `PARAKEET_NUM_THREADS`).
* **Block-parallel BGZF inflate fits inside `@par.jit`:** because BGZF splits
  `.bam` / `.fastq.gz` files into independent `≤ 64 KiB` blocks with known
  uncompressed sizes, [`examples/bgzf_and_streaming_io.py`](https://github.com/iskandr/parakeet/blob/master/examples/bgzf_and_streaming_io.py)
  compiles a full RFC-1951 DEFLATE inflater across all blocks in parallel. On
  20,000 FASTQ reads (6.31 MB raw, 97 blocks, Apple M1 Pro):
  - **Raw inflate:** Python `gzip` (1T) **226 MB/s** vs. Parakeet BGZF 1T
    **203 MB/s** vs. Parakeet BGZF multicore **1,579 MB/s** (**7.0×**).
  - **Decompress + FASTQ QC:** sequential `gzip` + QC **168 MB/s** vs.
    overlapped double-buffered `zlib` + QC **341 MB/s** (**2.0×**) vs. Parakeet
    parallel BGZF + QC **906 MB/s** (**5.4×**).
* **Use native codecs for single-stream `gzip`/`zstd`:** single-stream DEFLATE
  has a serial 32 KiB sliding-window dependency with no data parallelism—feed
  decompressed `uint8` slabs from `isa-l`/`libdeflate`/`zstd` into Parakeet.
