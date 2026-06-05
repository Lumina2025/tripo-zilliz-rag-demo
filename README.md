# Tripo x Zilliz Multimodal Asset Search

A demo app for searching Tripo-generated 3D render assets with text, reference images, metadata filters, and Zilliz Cloud vector search.

The UI is included in `static/` and is served by FastAPI. No separate frontend build step is required.

## What Is Included

```text
app.py             FastAPI web app and search API
tripo_rag.py       CLI for collection creation, embedding cache, import, and search
static/            Web UI, styles, JavaScript, and logo assets
.env.example       Environment variable template
requirements.txt   Python dependencies
```

This repository does not include private credentials, Feishu/blog drafts, logs, embedding cache files, or the full image dataset.

## Data Layout

Prepare your Tripo asset data in the project root:

```text
milvus_dataset.csv
milvus_render_images/
milvus_input_images/
```

`milvus_dataset.csv` must include at least these columns:

```text
project_id
caption
llm_keyword
llm_object
llm_category
llm_style
llm_color
llm_use_case
url
input_image_file
input_image_key
render_image_file
render_image_key
```

`milvus_render_images/` contains the render preview images that will be embedded and searched. `milvus_input_images/` is optional and is used only for displaying or submitting reference images.

## Model And Vector Field

- Embedding model: `google/gemini-embedding-2-preview` through OpenRouter
- Vector field: `multimodal_vector`
- Dimension: `3072`
- Metric: `COSINE`

The app embeds the Tripo render image directly. There is no captioning step.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=your-openrouter-api-key
ZILLIZ_TOKEN=your-zilliz-cloud-token
ZILLIZ_ENDPOINT=https://your-cluster-id.region.vectordb.zillizcloud.com:19537
ZILLIZ_DATABASE=default
TRIPO_MIN_SEARCH_SCORE=0.40
```

## Build The Index

Load environment variables:

```bash
set -a; source .env; set +a
```

Create the Zilliz Cloud collection:

```bash
python3 tripo_rag.py create-collection --drop-existing
```

Generate render-image embeddings and cache them locally:

```bash
python3 tripo_rag.py build-cache
```

Import records into Zilliz Cloud:

```bash
python3 tripo_rag.py import-data
```

Check collection stats:

```bash
python3 tripo_rag.py stats
```

## CLI Search

Text search:

```bash
python3 tripo_rag.py search \
  --text "fantasy sword with blue gemstone" \
  --top-k 5
```

Image search:

```bash
python3 tripo_rag.py search \
  --image milvus_input_images/example.png \
  --top-k 5
```

Text plus image search:

```bash
python3 tripo_rag.py search \
  --text "similar style game asset" \
  --image milvus_input_images/example.png \
  --top-k 5
```

## Web UI

Start the FastAPI app:

```bash
set -a; source .env; set +a
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The UI can search by:

- text only
- reference image only
- text plus reference image
- metadata filters such as category, style, and use case

`Top K` is the maximum number of results to show. `TRIPO_MIN_SEARCH_SCORE` filters out weak matches so the UI does not force-fill irrelevant results.

## Notes

- Do not commit `.env`.
- Do not commit large private datasets unless you intentionally want to publish them.
- If you deploy this app publicly, set environment variables in the deployment platform instead of committing secrets.
