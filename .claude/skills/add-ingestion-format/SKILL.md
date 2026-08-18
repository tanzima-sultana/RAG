---
name: add-ingestion-format
description: Adds ingestion/parsing support for a new document source format (PDF, scanned PDF, HTML, DOCX, TXT/MD, etc.) to this repo's src/ingestion/ module, with real sample-corpus testing. Use whenever the user asks to add support for a new file type to the RAG ingestion pipeline.
---

# Add ingestion format

`src/ingestion/` (`parsers.py`, `dedup.py`, `metadata.py`, `preprocessing.py`,
`ingestion.py`) is a format-dispatch ingestion layer that sits in front of the
existing parquet-based Wikipedia pipeline. TXT/MD support already exists there
(built against a pg19 sample corpus) — use it as the reference implementation
for the shape and conventions every new format should follow.

## The contract every format must satisfy

Downstream (`src/local/chunking.py:220-221`) only ever reads `doc['doc_id']`,
`doc['title']`, `doc['text']` from each document — a plain `dict`, not a
HuggingFace `Dataset`. Extra keys (`url`, `source_format`, ...) are harmless
and pass straight through. So a new format parser must produce dicts shaped
like:

```python
{
    "doc_id": ...,          # unique across the whole corpus
    "title": ...,
    "url": ...,             # None is fine if the format has no natural URL
    "text": ...,            # plain extracted text
    "source_format": ...,   # e.g. constants.PDF
}
```

`doc_id` must be unique and must not itself end in a way that breaks
`src/evaluation.py`'s `chunk_id.rsplit('_', 1)[0]` doc-id recovery — in
practice this is never actually a problem since `rsplit(maxsplit=1)` only ever
strips the single last `_<int>` chunk-index suffix that `chunking.py` appends,
regardless of how many underscores already exist in `doc_id`.

Module responsibilities (all in `src/ingestion/`):
- **`parsers.py`** — one `parse_<format>_file(file_path)` function per format,
  all reachable through `parse_file(file_path)`'s extension dispatch.
- **`preprocessing.py`** — format-agnostic text cleanup (whitespace
  normalization). Only add format-specific logic here if a format needs
  cleanup beyond what's already generic.
- **`dedup.py`** — exact dup via SHA-256, near-dup via 5-word shingle Jaccard
  similarity (threshold 0.8). Format-agnostic; don't touch unless a format's
  extracted text has systematically different characteristics that break this
  (see OCR note below).
- **`metadata.py`** — builds `{doc_id, title, url, source_format,
  ingestion_timestamp, content_hash}`. Format-agnostic, reuses
  `dedup.compute_content_hash`.
- **`ingestion.py`** — orchestrates parse → validate → preprocess → dedup →
  metadata for a whole directory. Its `parse_all_files` already catches any
  exception a parser raises and any empty-text result, logging both into
  `failure_records` instead of crashing the batch or silently dropping the
  file. New formats should raise on genuine failure (corrupt file, wrong
  encoding, missing library) and let `ingestion.py` isolate it — don't
  swallow errors inside the parser itself.

`constants.py` uses flat string constants grouped under a `# <Category>`
comment banner (see the existing `# Source format` block with `TXT`/`MD`) —
add one constant per new format there, not a nested enum/dict.

## Workflow

### 1. Confirm scope

If the user just says "add PDF support" with no detail, ask (or infer
sensibly and state the assumption) whether they mean text-based PDFs, scanned
(image-only) PDFs, or both — they need different libraries and the scanned
case needs OCR, which is much slower and lossier. See the per-format notes
below.

### 2. Get a real sample corpus — don't fabricate one

Testing against synthetic lorem-ipsum-style fixtures would have missed the
PG-19 lesson from the TXT/MD work: the pg19 dataset's "boilerplate fully
stripped" claim was only partially true — 21/25 real sample files still had a
leading `Produced by ...` line and 23/25 had a trailing `End of the Project
Gutenberg EBook of ...` line, discovered only by grepping actual downloaded
files. Every new format needs the equivalent step: pull a small (~25-file)
real sample corpus for that format before writing the parser, following
`scripts/download_pg19_sample.py` as the pattern (argparse script under
`scripts/`, writes to `data/raw/<format>_sample/`, which is already covered by
the repo's gitignore for `data/`). If a public source needs a new pip package
just to fetch it (e.g. `datasets` was needed for pg19), confirm the install
with the user first if it's not already in the `~/pyenv` environment.

### 3. Inspect sample files before writing the parser

Open/grep 2-3 real sample files and actually look for the format's typical
noise before assuming clean extraction — see the per-format gotchas below for
what to look for in each case. Write the boilerplate-stripping logic based on
what you actually find, not on the format's marketing claims about how clean
it is.

### 4. Install any new dependencies — confirm first

Check whether the library is already installed in `~/pyenv`
(`source ~/pyenv/bin/activate && python3 -c "import <lib>"`) before asking.
Some formats need a **system** package on top of the pip package (OCR needs
`tesseract-ocr` + `poppler-utils` via `apt`) — system-level installs need
`sudo` and are a more invasive/harder-to-reverse action than a pip install
into the user's own virtualenv, so always confirm those explicitly before
running them.

### 5. Add the constants.py entry

```python
# Source format
TXT="txt"
MD="md"
PDF="pdf"          # new
```

### 6. Write the parser and wire it into dispatch

Add `parse_<format>_file(file_path)` to `parsers.py` returning the doc-record
shape above, and extend `parse_file`'s extension check to route to it. Follow
this repo's code style (see `CLAUDE.md`): explicit if/elif branching, no
comprehensions/callbacks, self-descriptive names, no comments except where a
non-obvious constraint needs explaining (e.g. why a boilerplate regex exists).

### 7. Re-check dedup/preprocessing/metadata for format-specific assumptions

Usually nothing changes here. The one common exception is OCR'd text: OCR
noise (misread characters, broken line wraps) can suppress exact n-gram
shingle overlap between two documents that are genuinely near-duplicates, so
flag to the user if you think `dedup.py`'s 0.8 Jaccard threshold needs
loosening for a given format rather than silently changing the shared
threshold.

### 8. Build corrupt/dup fixtures from the real sample corpus

Mirror `tests/test_ingestion_txt.py`'s fixture: copy real sample files into a
`tmp_path` dir, then synthesize from them:
- an exact duplicate (byte-identical copy under a new filename)
- a near-duplicate (real file with a chunk of content removed)
- format-appropriate corruption cases (see per-format notes — a truncated
  file, an empty file, and at least one format-specific corruption such as a
  password-protected PDF or a docx with its zip container truncated)

### 9. Write tests/test_ingestion_<format>.py

Follow `tests/test_ingestion_txt.py`'s structure exactly:
- `pytestmark = pytest.mark.skipif(...)` guarding on the sample corpus
  existing on disk (so the suite doesn't hard-fail for someone who hasn't run
  the download script)
- assert every input file lands in exactly one of `kept_docs` /
  `duplicate_records` / `failure_records` (`len(kept) + len(dup) + len(fail)
  == total_input_files`) — this is the single invariant that proves nothing
  was silently dropped, regardless of which specific files you expect to fail
- assert the corrupt fixtures show up in `failure_records`, not just missing
  from `kept_docs`
- assert exact/near duplicates are caught, using an order-independent
  "duplicate family" check (don't assume which of two identical-content files
  sorts first and gets "kept" — `list_source_files` sorts alphabetically)
- assert every kept doc has a non-null 64-char `content_hash`

### 10. Run the test suite

```bash
source ~/pyenv/bin/activate
python3 -m pytest tests/test_ingestion_<format>.py -v
python3 -m pytest   # full suite, confirm no regressions
```

### 11. Run the pipeline end-to-end against the new corpus

Don't run this through `local_run.sh` / `scripts/build_rag.py` directly — that
would write into the repo's real `chunks/`, `embeddings/`, `index/`,
`manifests/` paths keyed by `dataset_size`, and could call the real Anthropic
API for eval-question generation and answer generation (real cost). Instead,
write a throwaway script (not a deliverable — don't commit it) that:
- `os.chdir`s into a scratch directory so all pipeline output lands outside
  the repo
- calls `ingest_directory(...)` to get docs, then drives
  `Chunking.compute_chunks` → `Embedding.generate_embeddings` →
  `Indexing.generate_faiss_index` / `generate_bm25_index` directly
- builds a tiny synthetic `eval_set` (`[{chunk_id, question, answer}]`) from a
  few real chunk texts rather than calling `src/eval_qa.py` (which needs
  Claude)
- calls `Retrieval(...retrieval_dense/bm25/hybrid...)` with `dry_run=True`
  (skips the real Anthropic answer-generation call, per
  `Retrieval.get_answers_batch`'s existing dry-run branch) and `reranking=0`
  (skips loading a cross-encoder, unless reranking itself is what you're
  validating)

Report explicitly if anything downstream turns out to assume Wikipedia-shaped
input (fixed numeric IDs, a particular text length distribution, etc.) — so
far chunking/embedding/indexing have had no such assumptions.

### 12. Show diffs before calling it done

Show the full diff/content of every new or changed file (constants.py,
parsers.py, any touched sibling ingestion module, the new test file, the new
download/sample script) before treating the work as finished, the same way
the TXT/MD ingestion work was reviewed.

## Per-format notes

**TXT/MD** (reference implementation, already done) — stdlib only, no new
dependency. Boilerplate stripped via regex on known residual patterns found
by grepping the real sample corpus, not assumed from the dataset's own
documentation.

**PDF (text-based/native)** — `pypdf` (`pip install pypdf`):
`pypdf.PdfReader(path).pages[i].extract_text()`, joined across pages. Watch
for: running headers/footers and page numbers repeating on every page
(inflates false near-dup shingle overlap if not stripped); multi-column
layouts extracting in an interleaved reading order; encrypted PDFs raising
`pypdf.errors.FileNotDecryptedError` (treat as a normal parse failure, isolate
via `ingestion.py`'s existing except-catch — don't add special-case recovery
logic for it).

**Scanned PDF (image-only, no text layer)** — try `pypdf` text extraction
first; if the extracted text is empty or near-empty, fall back to OCR via
`pytesseract` (`pip install pytesseract`, needs the `tesseract-ocr` system
binary) + `pdf2image` (`pip install pdf2image`, needs the `poppler-utils`
system binary) to rasterize each page before OCR'ing it. Confirm both system
packages with the user before `apt install`-ing them. OCR output is noisy
(misread characters, broken line wraps) — call this accuracy tradeoff out to
the user explicitly, and see the dedup note in step 7 above.

**HTML** — `beautifulsoup4` (`pip install beautifulsoup4`): strip
`<script>`/`<style>`/`<nav>`/`<header>`/`<footer>` before `.get_text()`. Real
HTML has far more non-content chrome than PG-19 had residual boilerplate
(cookie banners, ad slots, related-link sidebars, comment sections) — grepping
2-3 real sample pages matters even more here than it did for TXT. Consider
`trafilatura` (`pip install trafilatura`) as a purpose-built main-content
extractor if bs4 tag-stripping proves too lossy on real pages.

**DOCX** — `python-docx` (`pip install python-docx`):
`docx.Document(path)`, join `paragraph.text` across `document.paragraphs`.
Decide explicitly (and tell the user) whether `document.tables` cell text
should be included — don't silently drop table content. A docx is a zip
archive; a truncated/corrupted one raises `docx.opc.exceptions
.PackageNotFoundError` or `zipfile.BadZipFile` on open — isolate it the same
way as any other parse failure.
