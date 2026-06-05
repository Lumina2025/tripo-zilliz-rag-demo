import json
import tempfile
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tripo_rag import (
    COLLECTION_NAME,
    VECTOR_FIELD,
    OpenRouterEmbeddingClient,
    connect_client,
    filter_expr,
    require_env,
)


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
RENDER_DIR = ROOT / "milvus_render_images"
INPUT_DIR = ROOT / "milvus_input_images"
LOG_DIR = ROOT / "logs"
SEARCH_LOG = LOG_DIR / "search.log"
DEFAULT_MIN_SEARCH_SCORE = 0.4

app = FastAPI(title="Tripo Multimodal RAG Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger = logging.getLogger("tripo-ui")


def write_search_log(event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    with SEARCH_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def min_search_score() -> float:
    value = os.getenv("TRIPO_MIN_SEARCH_SCORE")
    if not value:
        return DEFAULT_MIN_SEARCH_SCORE
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid TRIPO_MIN_SEARCH_SCORE=%r; using %.2f", value, DEFAULT_MIN_SEARCH_SCORE)
        return DEFAULT_MIN_SEARCH_SCORE


class SearchArgs:
    def __init__(
        self,
        category: str = "",
        style: str = "",
        use_case: str = "",
    ):
        self.category = category or None
        self.style = style or None
        self.use_case = use_case or None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/render/{filename}")
def render_image(filename: str):
    path = (RENDER_DIR / filename).resolve()
    if not str(path).startswith(str(RENDER_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="Render image not found")
    return FileResponse(path)


@app.get("/input/{filename}")
def input_image(filename: str):
    path = (INPUT_DIR / filename).resolve()
    if not str(path).startswith(str(INPUT_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="Input image not found")
    return FileResponse(path)


def hit_to_result(rank: int, hit: dict) -> dict:
    entity = hit["entity"]
    render_file = entity.get("render_image_file", "")
    input_file = entity.get("input_image_file", "")
    return {
        "rank": rank,
        "score": hit["distance"],
        "render_url": f"/render/{render_file}" if render_file else "",
        "input_url": f"/input/{input_file}" if input_file else "",
        **entity,
    }


@app.post("/api/search")
async def search_assets(
    text: str = Form(""),
    category: str = Form(""),
    style: str = Form(""),
    use_case: str = Form(""),
    top_k: int = Form(12),
    image: Optional[UploadFile] = File(None),
):
    if not text.strip() and (image is None or not image.filename):
        raise HTTPException(status_code=400, detail="Provide text, image, or both.")

    logger.info(
        "search request text=%s image=%s category=%r style=%r use_case=%r top_k=%s",
        bool(text.strip()),
        bool(image and image.filename),
        category,
        style,
        use_case,
        top_k,
    )
    request_log = {
        "type": "request",
        "has_text": bool(text.strip()),
        "has_image": bool(image and image.filename),
        "image_filename": image.filename if image and image.filename else "",
        "category": category,
        "style": style,
        "use_case": use_case,
        "top_k": top_k,
        "text_preview": text.strip()[:120],
    }
    write_search_log(request_log)

    embedding_client = OpenRouterEmbeddingClient(require_env("OPENROUTER_API_KEY"))
    client = connect_client()

    image_path = None
    try:
        if image is not None and image.filename:
            suffix = Path(image.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await image.read())
                image_path = Path(tmp.name)

        if text.strip() and image_path:
            vector = embedding_client.embed_text_image(text.strip(), image_path)
        elif image_path:
            vector = embedding_client.embed_image(image_path)
        else:
            vector = embedding_client.embed_text(text.strip())

        args = SearchArgs(category, style, use_case)
        milvus_filter = filter_expr(args)
        logger.info("search filter expr=%r", milvus_filter)
        write_search_log({"type": "filter", "expr": milvus_filter})

        search_limit = max(1, min(top_k * 3, 50))
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            anns_field=VECTOR_FIELD,
            filter=milvus_filter,
            limit=search_limit,
            output_fields=[
                "project_id",
                "caption",
                "llm_keyword",
                "llm_object",
                "llm_category",
                "llm_style",
                "llm_color",
                "llm_use_case",
                "generation_mode",
                "render_image_file",
                "render_image_key",
                "input_image_file",
                "url",
            ],
            search_params={"metric_type": "COSINE"},
        )
    finally:
        if image_path and image_path.exists():
            image_path.unlink()

    min_score = min_search_score()
    raw_hits = list(results[0])
    filtered_hits = [hit for hit in raw_hits if float(hit["distance"]) >= min_score]
    limited_hits = filtered_hits[: max(1, min(top_k, 50))]

    raw_summary = [
        {
            "rank": rank,
            "score": round(float(hit["distance"]), 4),
            "category": hit["entity"].get("llm_category"),
            "style": hit["entity"].get("llm_style"),
            "object": hit["entity"].get("llm_object"),
            "render": hit["entity"].get("render_image_file"),
        }
        for rank, hit in enumerate(raw_hits, start=1)
    ]
    filtered_summary = [
        {
            "rank": rank,
            "score": round(float(hit["distance"]), 4),
            "category": hit["entity"].get("llm_category"),
            "style": hit["entity"].get("llm_style"),
            "object": hit["entity"].get("llm_object"),
            "render": hit["entity"].get("render_image_file"),
        }
        for rank, hit in enumerate(limited_hits, start=1)
    ]
    logger.info(
        "search results raw=%s filtered=%s min_score=%s",
        raw_summary,
        filtered_summary,
        min_score,
    )
    write_search_log(
        {
            "type": "results",
            "min_score": min_score,
            "raw_count": len(raw_hits),
            "filtered_count": len(limited_hits),
            "raw_summary": raw_summary,
            "summary": filtered_summary,
        }
    )

    return {
        "query": {"text": text, "has_image": bool(image and image.filename)},
        "min_score": min_score,
        "raw_count": len(raw_hits),
        "results": [hit_to_result(rank, hit) for rank, hit in enumerate(limited_hits, start=1)],
    }


@app.get("/api/health")
def health():
    try:
        client = connect_client()
        stats = client.get_collection_stats(COLLECTION_NAME)
        return {"ok": True, "collection": COLLECTION_NAME, "stats": stats}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
