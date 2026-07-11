# Document Upload Guide

Guidance for building a client's knowledge base. This reflects exactly what the
ingestion pipeline does today — not aspirational behavior.

## Supported formats

Only two file types are accepted. The frontend and backend both validate this and
reject anything else.

| Format | Extension | Use it for |
|--------|-----------|------------|
| **PDF**  | `.pdf`  | Manuals, guides, policies, brochures — narrative / prose content |
| **JSON** | `.json` | FAQs, package/plan catalogs, product data — structured Q&A |

No `.docx`, `.txt`, `.md`, `.csv`, `.xlsx`, or `.html`. Convert those to PDF (for
prose) or JSON (for structured data) first.

## Which format performs best

### JSON — best for FAQ / customer-care content (reach for this first)

Each entry is parsed as a discrete Q&A unit (`Q: … / A: …`), and the loader
auto-detects which fields are answer text vs. metadata. Because every FAQ becomes
its own clean chunk, retrieval is far more precise than pulling a paragraph out of a
wall of PDF text.

Recommended shape:

```json
{
  "items": [
    { "question": "How do I track my order?", "answer": "Go to My Orders and…", "category": "shipping" },
    { "question": "What are your business hours?", "answer": "We're open 9am–6pm…", "category": "general" }
  ]
}
```

- **Recognized text fields** (become the answer body): `question, answer,
  description, content, text, body, title, name, summary, details, benefits,
  features, includes, steps, rules, policy, procedure, instructions`.
- **Recognized metadata fields** (used for filtering, not shown as answer text):
  `id, type, category, tag, status, date, created, updated, author, priority,
  version, code, sku`.
- A top-level array (`[ {...}, {...} ]`), an object with an array field
  (`{ "items": [...] }`), or a single object are all accepted.

### PDF — best for long-form prose

Handbooks, terms & conditions, product guides. Works well **if** the PDF is:

- **Digitally generated** (exported from Word / Google Docs / InDesign), not scanned.
- **Text-based with clear headings** — the chunker detects section structure and
  splits on sentence boundaries.

## Can documents contain images? No.

PDF text is extracted with `pypdf`'s `extract_text()` — **text only, no OCR**.

- **Images are ignored entirely.** Diagrams, screenshots, photos, logos, and
  infographics contribute nothing to the knowledge base.
- **Scanned PDFs are effectively empty.** A PDF that is really a photo of a page
  yields little or no text and indexes as near-nothing. Run scanned sources through
  OCR (or Google Docs → export as PDF) first.

## Can documents contain tables? Partially.

Tables are not parsed as tables — `extract_text()` flattens them into a text stream.

- **Simple 2-column tables** (Feature → Value) usually survive as readable
  "label: value" lines and retrieve fine.
- **Complex / multi-column tables** (pricing matrices, comparison grids) often come
  out with columns interleaved or misaligned, so the model may misread which value
  belongs to which row.

**Recommendation:** for pricing tiers, plan comparisons, or anything tabular that
matters, model it as JSON (one object per plan/row) instead of relying on a PDF
table. Accuracy improves dramatically.

## How a document gets chunked

Text is split into **~600-token target chunks (800 hard max), with 100-token
overlap**, on sentence boundaries. Tips that improve retrieval:

- Use **real headings and short paragraphs** — the chunker respects structure.
- Keep each self-contained idea together; avoid one answer spanning many pages.
- Prefer **one topic per file** (e.g. `billing.pdf`, `returns.pdf`) over a single
  giant master PDF — it keeps chunks topically tight.
- After upload, the UI shows a **chunk preview**. Glance at it to confirm the text
  extracted cleanly — this is the fastest way to catch a scanned/garbled PDF.

## Note on language

Knowledge-base documents should be in **English**. Queries in Sinhala/Tamil are
translated to English for retrieval, and the answer is returned in the user's
language. Keeping the source documents in English gives the most reliable retrieval.

## Quick decision rule

- **FAQ / support answers / catalogs / pricing** → **JSON** (best accuracy).
- **Policies / manuals / narrative guides** → **digital PDF** with clear headings.
- **Scanned docs** → OCR to text first, then PDF.
- **Important tables / images** → re-express the data as JSON; images can't be read
  at all.
