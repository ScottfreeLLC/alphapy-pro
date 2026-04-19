# Codex — LLM-Native Knowledge Base System

## Design Document v0.1

---

## 1. What Is This

Codex is a personal knowledge base system where an LLM is the primary author, editor, and operator. Raw sources go in, a structured wiki comes out, and scheduled agents continuously maintain, enrich, and query it. The human steers; the LLM does the work.

**First domain:** Quantitative finance & trading — strategies, papers, market microstructure, backtesting methodologies, instruments, datasets, and tooling.

**Deployment target:** scottfreeserver (self-contained, containerized)

---

## 2. Core Concepts

### The Knowledge Loop

```
┌─────────────────────────────────────────────────┐
│                                                   │
│   INGEST ──► COMPILE ──► INDEX ──► QUERY          │
│     ▲                                 │            │
│     │         HEARTBEAT               │            │
│     │        (scheduled)              │            │
│     │           │                     │            │
│     │           ▼                     ▼            │
│     └────── LINT/ENRICH ◄──── FILE OUTPUTS ────┘  │
│                                                   │
└─────────────────────────────────────────────────┘
```

### Key Entities

| Entity | What it is |
|--------|-----------|
| **Source** | A raw document in `raw/` — PDF, markdown clip, CSV, image, code file, URL bookmark |
| **Article** | A compiled `.md` file in `wiki/` — authored by the LLM from one or more sources |
| **Index** | Auto-maintained files: master index, concept map, backlink registry, per-directory summaries |
| **Soul** | A markdown file defining the agent's identity, scope, style, and operating rules for this domain |
| **Job** | A defined task the agent can execute — compile, query, lint, enrich, report |
| **Heartbeat** | A scheduled sweep that checks for pending work and dispatches jobs |

---

## 3. Directory Structure

```
codex/
├── config/
│   ├── soul.md                  # Agent identity + operating rules
│   ├── settings.yaml            # LLM provider config, paths, schedules
│   └── providers/               # LLM provider adapters
│       ├── anthropic.py
│       ├── openai.py
│       └── base.py
│
├── domains/
│   └── quant-finance/           # Each domain is a self-contained KB
│       ├── soul.md              # Domain-specific soul (inherits from root)
│       ├── raw/                 # Unprocessed source material
│       │   ├── papers/
│       │   ├── articles/
│       │   ├── datasets/
│       │   ├── code/
│       │   └── images/
│       ├── wiki/                # LLM-compiled knowledge base
│       │   ├── _index.md        # Master index (auto-maintained)
│       │   ├── _concepts.md     # Concept map with relationships
│       │   ├── _backlinks.md    # Reverse link registry
│       │   ├── _sources.md      # Source manifest with status
│       │   ├── concepts/        # Core concept articles
│       │   ├── strategies/      # Trading strategy deep-dives
│       │   ├── instruments/     # Asset class / instrument profiles
│       │   ├── methods/         # Quantitative methods & techniques
│       │   ├── tools/           # Software, libraries, platforms
│       │   └── people/          # Key figures, researchers, funds
│       ├── outputs/             # Query results, reports, visualizations
│       │   ├── reports/
│       │   ├── slides/
│       │   └── charts/
│       └── .meta/               # Internal metadata
│           ├── compile_log.jsonl # What was compiled, when, from what
│           ├── heartbeat_log.jsonl
│           └── source_hashes.json # Dedup / change detection
│
├── engine/
│   ├── __init__.py
│   ├── compiler.py              # Raw → Wiki compilation pipeline
│   ├── indexer.py               # Index maintenance (master, concepts, backlinks)
│   ├── heartbeat.py             # Scheduled sweep + job dispatch
│   ├── query.py                 # Q&A engine over the wiki
│   ├── linter.py                # Health checks, contradiction detection
│   ├── enricher.py              # Web search, gap filling, connection discovery
│   ├── ingest.py                # Source ingestion + normalization
│   └── renderer.py              # Output formatting (md, marp slides, matplotlib)
│
├── tools/
│   ├── search.py                # CLI search over wiki (BM25 + semantic)
│   ├── ask.py                   # CLI Q&A interface
│   ├── compile.py               # CLI trigger compilation
│   ├── lint.py                  # CLI trigger health check
│   ├── ingest.py                # CLI add new source(s)
│   └── report.py                # CLI generate report/slides
│
├── web/                         # Lightweight status/search UI (optional, phase 2)
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 4. The Soul File

The soul is what makes this more than a script. It's a markdown file the LLM reads before every operation.

```markdown
# Soul: Quant Finance Research Agent

## Identity
You are a quantitative finance research analyst maintaining a knowledge base
for Mark Scott at Scottfree Analytics. Your job is to compile, organize,
cross-reference, and surface insights from raw research materials.

## Scope
- Quantitative trading strategies (systematic, statistical arbitrage, market making)
- Market microstructure and execution
- Time series analysis and forecasting
- Portfolio construction and risk management
- Backtesting methodology and pitfalls
- Relevant tools, datasets, and infrastructure

## Style
- Write like a senior quant explaining to a peer, not a textbook
- Be opinionated — flag when a paper's methodology is weak
- Cross-reference aggressively — every article should link to related concepts
- Mark confidence levels: [HIGH] sourced from primary data, [MED] synthesized,
  [LOW] speculative or single-source

## Operating Rules
- Never delete or overwrite raw sources
- Always log what you compile and why in .meta/compile_log.jsonl
- When you find contradictions, create a dedicated article documenting both sides
- Prefer depth over breadth — a 2000-word article on one concept beats
  ten 200-word stubs
- When uncertain, say so. Don't hallucinate citations.
```

---

## 5. Component Design

### 5.1 Ingest (`engine/ingest.py`)

**Job:** Normalize raw sources into a consistent format the compiler can process.

- Accepts: PDF, .md, .html, .csv, .py, .ipynb, images, URL bookmarks
- Outputs: Normalized `.md` in `raw/` with frontmatter metadata
- Deduplication via content hashing (stored in `.meta/source_hashes.json`)
- For URLs: fetch + convert to markdown (similar to Obsidian Web Clipper)
- For PDFs: extract text + images, store images alongside
- For code: wrap in fenced blocks with language tags
- Registers new sources in `wiki/_sources.md` with status `PENDING`

```yaml
# Frontmatter added to every normalized source
---
source_id: src_20260403_001
title: "Market Microstructure Invariance"
source_type: paper
original_format: pdf
ingested_at: 2026-04-03T14:30:00Z
url: https://arxiv.org/abs/...
tags: [microstructure, invariance, volume-clock]
status: pending  # pending | compiled | stale | superseded
hash: sha256:abc123...
---
```

### 5.2 Compiler (`engine/compiler.py`)

**Job:** Read pending sources, produce or update wiki articles.

This is the most LLM-intensive component. For each pending source:

1. Read the soul file for voice/scope/rules
2. Read `_index.md` and `_concepts.md` for existing knowledge context
3. Read the source document
4. Decide: new article(s), or update to existing article(s)?
5. Draft the article(s) with proper frontmatter, backlinks, confidence tags
6. Update all affected index files
7. Log the operation to `compile_log.jsonl`
8. Mark source status as `compiled`

**Article frontmatter:**

```yaml
---
article_id: art_market_microstructure_invariance
title: "Market Microstructure Invariance"
category: concepts
created_at: 2026-04-03T15:00:00Z
updated_at: 2026-04-03T15:00:00Z
sources: [src_20260403_001, src_20260312_017]
confidence: HIGH
related: [art_volume_clocks, art_kyle_model, art_optimal_execution]
tags: [microstructure, invariance, volume-clock, kyle-model]
---
```

**Key compiler behaviors:**
- If a source touches multiple concepts, produce multiple articles
- If a source contradicts an existing article, create a `_contradictions/` entry
- Summaries at the top of each article (2-3 sentences), detail below
- Every claim linked back to source_id where possible
- Inter-article links use standard `[[wiki-link]]` syntax for Obsidian compat

### 5.3 Indexer (`engine/indexer.py`)

**Job:** Maintain the auto-generated index files that make the wiki navigable without embeddings.

Files maintained:
- `_index.md` — Master table of all articles, sorted by category, with one-line summaries
- `_concepts.md` — Concept graph as a structured list showing relationships
- `_backlinks.md` — For each article, what other articles link to it
- `_sources.md` — All raw sources with their compilation status
- Per-directory `_readme.md` — Summary of what's in each subdirectory

**This is the poor-man's RAG.** The LLM reads `_index.md` first on every query to decide which articles to pull into context. At ~100-500 articles, this works. Beyond that, we layer in the search tool.

### 5.4 Heartbeat (`engine/heartbeat.py`)

**Job:** Wake up on schedule, survey the KB, dispatch work.

Checks (configurable, default every 30 min):

1. **New sources?** — Any files in `raw/` not yet in `_sources.md` → trigger ingest
2. **Pending compilation?** — Any sources with status `pending` → trigger compiler
3. **Stale articles?** — Any articles older than threshold with newer related sources → flag for recompilation
4. **Index drift?** — Any articles added/modified since last index rebuild → trigger indexer
5. **Lint alerts?** — Run lightweight contradiction/consistency checks → log findings
6. **Enrichment candidates?** — Articles with `[LOW]` confidence or `TODO` markers → queue for enrichment

Implementation: `cron` or `systemd` timer in Docker, calls `heartbeat.py` which dispatches to other engines.

### 5.5 Query (`engine/query.py`)

**Job:** Answer questions against the wiki.

Flow:
1. Read soul + `_index.md`
2. Identify relevant articles from index (LLM decides, optionally augmented by search tool)
3. Pull relevant articles into context
4. Generate answer with citations back to articles/sources
5. Optionally file the answer as a new article or output

Modes:
- **Direct Q&A** — "What are the main approaches to optimal execution?"
- **Synthesis** — "Compare momentum strategies across the three papers I ingested last week"
- **Gap analysis** — "What topics in my wiki have the thinnest coverage?"
- **Connection discovery** — "What unexpected links exist between my market microstructure and time series articles?"

### 5.6 Linter (`engine/linter.py`)

**Job:** Health checks over the wiki.

Checks:
- Orphan articles (no backlinks)
- Broken wiki-links
- Contradictory claims across articles (LLM-assisted)
- Missing categories or tags
- Articles with no source attribution
- Stale content (old articles, newer data available)
- Confidence distribution (too many `[LOW]` articles = research gaps)

Output: `outputs/reports/lint_report_YYYYMMDD.md`

### 5.7 Enricher (`engine/enricher.py`)

**Job:** Fill gaps using web search and external data.

- Takes `[LOW]` confidence articles or explicit enrichment requests
- Searches the web for corroborating/contradicting data
- Downloads and ingests new sources into `raw/`
- Triggers recompilation of affected articles
- Logs all enrichment actions for auditability

### 5.8 Renderer (`engine/renderer.py`)

**Job:** Convert wiki content into other formats.

- **Markdown** — default, already native
- **Marp slides** — for presentation-style output
- **Matplotlib/charts** — for quantitative visualizations
- **HTML reports** — for richer formatting
- All outputs go to `outputs/` and can be filed back into wiki

---

## 6. LLM Provider Abstraction

```python
# config/providers/base.py
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict], system: str = "",
                       max_tokens: int = 4096, temperature: float = 0.3) -> str:
        ...

    @abstractmethod
    async def complete_structured(self, messages: list[dict], schema: dict,
                                   system: str = "") -> dict:
        ...
```

Settings in `config/settings.yaml`:

```yaml
llm:
  default_provider: anthropic
  providers:
    anthropic:
      model: claude-sonnet-4-20250514
      api_key_env: ANTHROPIC_API_KEY
    openai:
      model: gpt-4o
      api_key_env: OPENAI_API_KEY

  # Route different tasks to different providers/models
  routing:
    compile: anthropic       # Needs best writing quality
    index: anthropic         # Cheaper, structured output
    query: anthropic         # Needs best reasoning
    lint: anthropic          # Pattern matching, cheaper ok
    enrich: anthropic        # Web search integration
```

---

## 7. CLI Interface

```bash
# Ingest
codex ingest paper.pdf                    # Add a single source
codex ingest ./batch/                     # Add a directory of sources
codex ingest --url https://arxiv.org/...  # Fetch and ingest URL

# Compile
codex compile                             # Compile all pending sources
codex compile --source src_20260403_001   # Compile specific source
codex compile --rebuild art_kyle_model    # Recompile specific article

# Query
codex ask "What are the main momentum strategies in my KB?"
codex ask --synthesis "Compare all market making approaches"
codex ask --gaps                          # What's thin?
codex ask --connections                   # Surprise me

# Search
codex search "volume clock"              # Keyword + semantic search
codex search --tag microstructure         # Tag-based filtering

# Maintenance
codex lint                                # Run health checks
codex enrich art_kyle_model               # Enrich specific article
codex heartbeat                           # Run one heartbeat cycle manually

# Output
codex report "Weekly quant research summary" --format marp
codex report "Strategy comparison" --format html

# Status
codex status                              # KB stats, pending work, recent activity
```

---

## 8. Deployment (Docker on scottfreeserver)

```yaml
# docker-compose.yml
services:
  codex:
    build: .
    volumes:
      - ./domains:/app/domains        # KB data persists on host
      - ./config:/app/config          # Config persists on host
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8420:8420"                   # Web UI (phase 2)

  # Optional: cron container for heartbeat
  heartbeat:
    build: .
    command: ["python", "-m", "engine.heartbeat", "--daemon"]
    volumes:
      - ./domains:/app/domains
      - ./config:/app/config
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

---

## 9. Build Phases

### Phase 1: Foundation (build now)
- [ ] Directory structure + config
- [ ] LLM provider abstraction (Anthropic first)
- [ ] Soul file for quant finance
- [ ] Ingest pipeline (PDF, markdown, URL)
- [ ] Compiler (source → wiki article)
- [ ] Indexer (auto-maintain _index.md, _concepts.md, _backlinks.md)
- [ ] CLI: `ingest`, `compile`, `status`

### Phase 2: Query + Maintenance
- [ ] Query engine (Q&A, synthesis, gaps)
- [ ] Search tool (BM25 + optional semantic)
- [ ] Linter (health checks)
- [ ] CLI: `ask`, `search`, `lint`

### Phase 3: Automation
- [ ] Heartbeat daemon
- [ ] Enricher (web search gap-filling)
- [ ] Renderer (Marp slides, matplotlib, HTML)
- [ ] CLI: `heartbeat`, `enrich`, `report`

### Phase 4: Scale + UI
- [ ] Web UI for search + status
- [ ] Multi-domain support (add domains beyond quant finance)
- [ ] MCP server interface (expose KB as tools for external agents)
- [ ] Obsidian plugin or sync

---

## 10. Open Questions

1. **Naming**: "Codex" works but is overloaded (OpenAI used it). Alternatives: `Vault`, `Loom`, `Folio`, `Grimoire`, `Ledger`?
2. **Obsidian compatibility**: Do we optimize wiki link syntax and directory structure for direct Obsidian viewing from day one?
3. **Git integration**: Should the wiki be a git repo with auto-commits after each compilation? Gives you full history + diff.
4. **Token budget management**: How aggressive should the compiler be about staying within context limits? Pre-chunk large sources, or trust the LLM to handle it?
5. **Multi-agent vs. single-agent**: Start with one agent that wears many hats, or immediately split into compiler/querier/linter roles à la Claire Vo?

---

## 11. Why This Isn't Just Another RAG App

- **The LLM writes the knowledge base, not just reads it.** Most RAG systems index existing docs and retrieve chunks. This system *compiles new artifacts* from raw sources.
- **The wiki is human-readable.** You can open it in Obsidian and browse it without any AI. It's useful even if the LLM is off.
- **Outputs feed back in.** Every query that produces a report can be filed back, making the KB smarter over time. This is a flywheel, not a pipeline.
- **The soul file means the KB has a point of view.** It's not neutral retrieval — it's opinionated synthesis scoped to your domain and preferences.
- **Scheduled maintenance means it stays healthy.** Staleness, contradictions, and gaps get caught automatically instead of rotting silently.
