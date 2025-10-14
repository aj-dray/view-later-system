# Later System

**Description:** A Next.js web app and FastAPI server for storing and labeling online articles for later reading. Primarily for personal use and experimentation, but provides a foundation for a deployable product.

**Completion Date:** Work in Progress.

---

## Key Features
- Save, summarise, and embed web pages, youtube videos and pdfs from URLs (currently does not support paywalled content or uploaded files).
- Link to gmail to automatically add newsletters as new items.
- Organise and order items with status tracking.
- Cluster items based on embedding vectors with automated text labels.
- Lexical, semantic and agentic search using chunk embeddings.
- MCP server for Claude Desktop integration (queue management, search, content operations).

---

## Motivation & Vision

As AI-generated summaries increasingly homogenise information delivery, I wanted to build a framework that encourages the consumption of content in its organic, authored manifestation. I believe there is still value to manually grappling with the diverse textures of raw human communication without risk of intent and nuance being denoised by regeneration.

Yet, finding time to read is challenging. I frequently encounter content I intend to consume; I infrequently consume it. Tools like Obsidian or Notion allow content clipping and storage, but lack autonomy over analysis and interaction. The objective is to develop a ground-truth "inbox" for read-it-later online content that lowers the activation energy for engagement through smart priority ordering and creative visualisation.

---

## Current Implementation

This project is a step towards this objective, providing a service that accepts web article URLs, which are scraped, vector embedded, and stored. The monorepo is structure as follows:

```
.
├─ frontend/        → Next.js web app
├─ backend/         → FastAPI service for database access and ML services
│  └─ aglib/        → Lightweight Python package for building agents on top of LiteLLM
├─ database/        → Volumes and Docker container for PostgreSQL (local development)
└─ mcp-server/      → Model Context Protocol server for Claude Desktop integration
```

### Frontend

This provides simple interface for interacting with and visualising the data. I've streamlined the UI my own personal use; I need to redesign to make interaction more intuitive for users without any context.

Note that parts of the frontend have been implemented quickly, lacking the modular, clean development style React encourages. Refactoring is needed for maintainability and to better support Next.js server/client caching. Frontend caching of items is also needed to prevent reclustering and re-ranking on tab reloads.

Note that some areas of the frontend have been implemented crudely, neglecting obvious opportunities to leverage the modular design patterns that React development compels (coding agents feel no such compulsion). I therefore plan to refactor these sections, both for improved maintainability and to better support Next.js server/client caching strategies. I also need to implement frontend caching to prevent the inefficiency of repeated querying, reclustering, and re-ranking on each page change.

### Backend

When parsing a URL, three key steps occur:
1. **Extract:** Receive URL and extract metadata and content using Trafilatura
2. **Summarize:** Provide content and metadata to an LLM to generate a short summary and define an "expiry score" (0–1.0) for queue ranking, where higher values indicate faster decay in relevance
3. **Embed:** Use an embedding model (mistral-embed) to create 1024-dimensional vectors for full content and overlapping chunks. PostgreSQL also creates standard text-search embeddings. If context exceeds limits, the mean of chunk embeddings is used.

Backend services include:
* **Dimensional Reduction:** PCA, [t-SNE](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf), and [UMAP](https://arxiv.org/abs/1802.03426) for 2D visualization of embedding space
* **Clustering:** K-means, hierarchical clustering (agglomerative), and DBSCAN
* **Cluster Labeling:** Passes summaries from each cluster to an LLM (mistral-medium-2508) for label generation
* **Search:** PostgreSQL native text search ("lexical") or "semantic" cosine similarity between embedded queries (IVFFLAT indexed via pgvector) and embeddings, with Cohere's cross-encoder reranking and threshold filtering. Agentic search uses LLM to augment and iterate on embedding queries based on retrieved results – it is effectively a RAG system, but objective is still to return items rather than produce the "augmented generation". Agent reasoning is streamed to frontend. See "Future Developments" for steps towards full natural language search.

For more information on aglib, see [its repo](https://github.com/aj-dray/aglib).

### Database

Postgres is database is used. Clearly overkill for this project, but lays the foundation for a scalable product and is equipped with useful extensions such as vector indexing and serach that are well-suited to this service.

---

## Quick Start

### Setup

You can test the web app I've hosted at [later-system-frontend.vercel.app](https://later-system-frontend.vercel.app/).

Alternatively, you can run locally. By default Next.js app runs locally at http://localhost:3000. There are two simple ways to run locally:

```bash
# Export the required API keys (see backend/.env.example) in docker-compose (#1) or .env file (#2)

## Option 1 – Docker Compose (production mode)
docker compose up --build

## Option 2 – Development
# Database
docker-compose up postgres  # or use alternative PostgreSQL hosting

# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e aglib
./run.sh


# Frontend
cd frontend
./run.sh
```

### Authentication

On the "Sign Up" page, select "Request Demo Login" to get login credentials for an account preseeded with example data. Demo accounts requests are rate-limited to three per day per IP. Note: the local pre-seeded database does not contain chunk embeddings and so chunk-level search is note possible.

#### Access Token Endpoint

The backend exposes a simple endpoint to exchange a username and password for a JWT access token.

- **Route:** `POST /user/access-token`
- **Body:** JSON object with the following fields
  - `username` (string, required): account username
  - `password` (string, required): account password
  - `expires_in_hours` (integer, optional): override the default non-expiring token lifetime; must be greater than 0
- **Response:** JSON containing `access_token` and `token_type` (`bearer`). Tokens are stateless JWTs signed with the `BACKEND_SECRET`; no server-side storage occurs, so clients must persist and present the token (for example via the `Authorization: Bearer <token>` header).

### Navigating the Interface

The **navigation panel** (top right) provides access to three pages (see next session). From left to right, the available pages are:
* **Queue Page:** List of articles that can be filtered and ordered. The "priority" ordering uses a sigmoid function where `priority = 1 / (1 + exp(-k * ((days_since_added * expiry_score / base_period) - 0.5)))` (k=5, base_period=3 days). Higher expiry scores rise to the top within 3 days and stays there until dismissed.
* **Graph Page:** Interactive 2D plot of dimensionally reduced embeddings with cluster labels. Pulsing, moving nodes indicate dimensional reduction is processing. The box below the control panel shows color-coded cluster labels (pulse while LLM generates labels). Click a node to freeze its item card for interaction
* **Search Page:** Perform semantic and lexical search. Note: semantic search is similarity-based, not LLM-powered (yet).

The **control panel** (below navigation) contains page-specific settings: click controls to reveal dropdowns or textboxes – page content refreshes automatically on changes (~1s delay).

On each page ttems display as cards with metadata (hover and click on graph nodes to see the card). The **status** indicator (top-right corner) changes on hover:
* **Queued** (blue, default): ranked on queue page
* **Bookmarked** (purple): for active reading
* **Paused**: excluded from queue ranking until manually requeued
* **Deleted** (red): removed from database
* **Completed** (green): persists in database

**Adding Items:** Paste URLs into the white add panel (bottom right, queue page only). The top of the page animates as metadata and summary stream from the backend. On completion, items move into their queue position. Multiple URLs can be queued simultaneously. Very large items may fail due to memory limits on free-tier servers – errors appear on pending cards and must be dismissed manually. Retrying often succeeds.

**Log Out:** Click the user panel with your username in the bottom corner.

---

## Further Development

Key areas for improvement:

- Expand scraping to handle videos, PDFs, and podcasts
- Integrate browser-based scraping (i.e. web extension) using session cookies to access anti-bot sites and paywalled content. The MCP server provides a foundation for integration with [existing](https://www.perplexity.ai/comet) (and [future](https://www.theverge.com/news/704162/opeani-ai-web-browser-chatgpt)) agentic browser systems to collect and return data to the service, making stored data accessible through future agentic interfaces
- Enhance queue prioritization using clustering data, past interactions, and additional context (e.g., serving related articles sequentially, learning from successfully read content). Additionally LLM agent could evaluate queuing decisions to identify algorithmic shortcomings
- Use k-means clustering of chunk embeddings (rather than single full-text embeddings) for multi-topic articles and more nuanced clustering
- Automate clustering parameter selection using evaluation methods like silhouette score and Calinski-Harabasz index
- Improve agentic to simulatenously perform multiple retrieval queries and combine answers rather than synchronous iterating. Also add planning node to allow handling of more complex/multi-topic queries. Add capabilities for search to use SQL queries where appropriate for filtering by metadata rather than semantic.
