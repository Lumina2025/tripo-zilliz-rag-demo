#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import time
from pathlib import Path

import requests
from pymilvus import DataType, MilvusClient


COLLECTION_NAME = "tripo_multimodal_assets"
MODEL_NAME = "google/gemini-embedding-2-preview"
VECTOR_FIELD = "multimodal_vector"
# Use the model's default full-size embedding. Keeping this fixed matters because
# Milvus/Zilliz vector field dimensions are immutable after collection creation.
VECTOR_DIM = 3072

DATASET_PATH = Path("milvus_dataset.csv")
RENDER_IMAGE_DIR = Path("milvus_render_images")
# Cache stores expensive OpenRouter embeddings keyed by project_id. It is safe
# to resume build-cache after interruption because cached rows are skipped.
EMBEDDING_CACHE_PATH = Path("cache/openrouter_gemini_multimodal_embeddings.jsonl")

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def get_zilliz_endpoint() -> str:
    return require_env("ZILLIZ_ENDPOINT")


def get_zilliz_database() -> str:
    return os.getenv("ZILLIZ_DATABASE", "default")


def l2_normalize(vector: list[float]) -> list[float]:
    # COSINE search does not require pre-normalization in Milvus, but normalizing
    # keeps cached vectors consistent and makes local score debugging easier.
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def data_url_for_image(path: Path) -> str:
    # OpenRouter accepts local images through base64 data URLs, so render images
    # do not need to be uploaded to public object storage first.
    mime_type = mimetypes.guess_type(path.name)[0] or "image/webp"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


class OpenRouterEmbeddingClient:
    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/zilliz/tripo-rag-demo",
            "X-Title": "tripo-rag-demo",
        }

    def embed_content(self, content: list[dict]) -> list[float]:
        # Important: OpenRouter embeddings image input requires each input item
        # to wrap multimodal content blocks as {"content": [...]}. Passing
        # [{"type": "image_url", ...}] directly under input returns a schema 400.
        body = {
            "model": self.model,
            "input": [{"content": content}],
            "encoding_format": "float",
        }
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers=self._headers(),
            json=body,
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:1200]}")
        payload = response.json()
        try:
            vector = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {payload}") from exc
        if len(vector) != VECTOR_DIM:
            raise RuntimeError(f"Expected dim={VECTOR_DIM}, got dim={len(vector)}")
        return l2_normalize(vector)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_content([{"type": "text", "text": text}])

    def embed_image(self, image_path: Path) -> list[float]:
        # The indexed asset representation is the Tripo render output, not the
        # optional user input/reference image.
        return self.embed_content(
            [{"type": "image_url", "image_url": {"url": data_url_for_image(image_path)}}]
        )

    def embed_text_image(self, text: str, image_path: Path) -> list[float]:
        return self.embed_content(
            [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_url_for_image(image_path)}},
            ]
        )


def read_rows(limit: int | None = None) -> list[dict]:
    with DATASET_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    return rows


def generation_mode(row: dict) -> str:
    # input_image is stored as metadata only; it is useful for filtering/debugging
    # whether an asset came from text-to-3D or image/text-to-3D.
    return "image_to_3d" if row.get("input_image_file") else "text_to_3d"


def load_embedding_cache(path: Path = EMBEDDING_CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Ignore stale cache rows from a different model/dimension so they
            # cannot be accidentally inserted into an incompatible collection.
            if item.get("model") == MODEL_NAME and item.get("dim") == VECTOR_DIM:
                cache[item["project_id"]] = item
    return cache


def append_embedding_cache(project_id: str, vector: list[float], path: Path = EMBEDDING_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "project_id": project_id,
        "model": MODEL_NAME,
        "dim": VECTOR_DIM,
        "source": "render_image",
        VECTOR_FIELD: vector,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def connect_client() -> MilvusClient:
    return MilvusClient(
        uri=get_zilliz_endpoint(),
        token=require_env("ZILLIZ_TOKEN"),
        db_name=get_zilliz_database(),
    )


def create_collection(args) -> None:
    client = connect_client()
    collection = args.collection
    if client.has_collection(collection):
        if not args.drop_existing:
            print(f"Collection already exists: {collection}")
            return
        client.drop_collection(collection)
        print(f"Dropped existing collection: {collection}")

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("project_id", DataType.VARCHAR, max_length=64)
    schema.add_field("operator_id", DataType.VARCHAR, max_length=64)
    schema.add_field("caption", DataType.VARCHAR, max_length=2048)
    schema.add_field("llm_keyword", DataType.VARCHAR, max_length=512)
    schema.add_field("llm_object", DataType.VARCHAR, max_length=512)
    schema.add_field("llm_category", DataType.VARCHAR, max_length=128)
    schema.add_field("llm_style", DataType.VARCHAR, max_length=128)
    schema.add_field("llm_color", DataType.VARCHAR, max_length=256)
    schema.add_field("llm_use_case", DataType.VARCHAR, max_length=128)
    schema.add_field("generation_mode", DataType.VARCHAR, max_length=32)
    schema.add_field("url", DataType.VARCHAR, max_length=512)
    schema.add_field("input_image_file", DataType.VARCHAR, max_length=256)
    schema.add_field("input_image_key", DataType.VARCHAR, max_length=512)
    schema.add_field("render_image_file", DataType.VARCHAR, max_length=256)
    schema.add_field("render_image_key", DataType.VARCHAR, max_length=512)
    # Single unified multimodal vector field. Text, image, and text+image query
    # embeddings are all generated by the same OpenRouter model and searched here.
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
    print(f"Created collection: {collection}")


def build_embedding_cache(args) -> None:
    embedding_client = OpenRouterEmbeddingClient(require_env("OPENROUTER_API_KEY"))
    rows = read_rows(args.limit)
    cache = load_embedding_cache()
    remaining = [row for row in rows if row["project_id"] not in cache]
    print(f"Rows: {len(rows)}; cached: {len(cache)}; remaining: {len(remaining)}")

    for index, row in enumerate(remaining, start=1):
        image_path = RENDER_IMAGE_DIR / row["render_image_file"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        # Index only the final 3D render preview. input_image_file is stored in
        # metadata but intentionally not used as the primary retrieval vector.
        vector = embedding_client.embed_image(image_path)
        append_embedding_cache(row["project_id"], vector)
        print(f"[{index}/{len(remaining)}] embedded render image {row['project_id']} {row['render_image_file']}")
        if args.sleep:
            time.sleep(args.sleep)


def row_to_entity(row_index: int, row: dict, cached_item: dict) -> dict:
    return {
        "id": row_index,
        "project_id": row.get("project_id", ""),
        "operator_id": row.get("operator_id", ""),
        "caption": row.get("caption", ""),
        "llm_keyword": row.get("llm_keyword", ""),
        "llm_object": row.get("llm_object", ""),
        "llm_category": row.get("llm_category", ""),
        "llm_style": row.get("llm_style", ""),
        "llm_color": row.get("llm_color", ""),
        "llm_use_case": row.get("llm_use_case", ""),
        "generation_mode": generation_mode(row),
        "url": row.get("url", ""),
        "input_image_file": row.get("input_image_file", ""),
        "input_image_key": row.get("input_image_key", ""),
        "render_image_file": row.get("render_image_file", ""),
        "render_image_key": row.get("render_image_key", ""),
        VECTOR_FIELD: cached_item[VECTOR_FIELD],
    }


def import_data(args) -> None:
    client = connect_client()
    if not client.has_collection(args.collection):
        raise SystemExit(
            f"Collection does not exist: {args.collection}. "
            f"Run: python3 tripo_rag.py create-collection"
        )
    rows = read_rows(args.limit)
    cache = load_embedding_cache()
    entities = []
    for row_index, row in enumerate(rows):
        cached_item = cache.get(row["project_id"])
        if cached_item is None or VECTOR_FIELD not in cached_item:
            raise SystemExit(
                f"Missing embedding for project_id={row['project_id']}. "
                f"Run: python3 tripo_rag.py build-cache"
            )
        entities.append(row_to_entity(row_index, row, cached_item))

    total = 0
    for start in range(0, len(entities), args.batch_size):
        batch = entities[start : start + args.batch_size]
        result = client.insert(collection_name=args.collection, data=batch)
        total += result.get("insert_count", len(batch))
        print(f"Inserted {total}/{len(entities)}")
    client.flush(args.collection)
    print(f"Imported {total} rows into {args.collection}")


def filter_expr(args) -> str:
    filters = []
    if args.category:
        filters.append(f'llm_category == "{args.category}"')
    if args.style:
        filters.append(f'llm_style == "{args.style}"')
    if args.use_case:
        filters.append(f'llm_use_case == "{args.use_case}"')
    generation_mode = getattr(args, "generation_mode", None)
    if generation_mode:
        filters.append(f'generation_mode == "{generation_mode}"')
    return " and ".join(filters)


def search(args) -> None:
    if not args.text and not args.image:
        raise SystemExit("Provide --text, --image, or both.")

    embedding_client = OpenRouterEmbeddingClient(require_env("OPENROUTER_API_KEY"))
    client = connect_client()
    if args.text and args.image:
        # OpenRouter creates one joint embedding for text+image in the same
        # vector space as render_image embeddings.
        vector = embedding_client.embed_text_image(args.text, Path(args.image))
    elif args.image:
        vector = embedding_client.embed_image(Path(args.image))
    else:
        vector = embedding_client.embed_text(args.text)

    results = client.search(
        collection_name=args.collection,
        data=[vector],
        anns_field=VECTOR_FIELD,
        filter=filter_expr(args),
        limit=args.top_k,
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
    for rank, hit in enumerate(results[0], start=1):
        entity = hit["entity"]
        print(json.dumps({"rank": rank, "score": hit["distance"], **entity}, ensure_ascii=False, indent=2))


def stats(args) -> None:
    client = connect_client()
    print(json.dumps(client.get_collection_stats(args.collection), indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Tripo multimodal RAG demo on Zilliz Cloud.")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-collection")
    create.add_argument("--drop-existing", action="store_true")
    create.set_defaults(func=create_collection)

    cache = subparsers.add_parser("build-cache")
    cache.add_argument("--limit", type=int)
    cache.add_argument("--sleep", type=float, default=0.0)
    cache.set_defaults(func=build_embedding_cache)

    ingest = subparsers.add_parser("import-data")
    ingest.add_argument("--limit", type=int)
    ingest.add_argument("--batch-size", type=int, default=100)
    ingest.set_defaults(func=import_data)

    search_cmd = subparsers.add_parser("search")
    search_cmd.add_argument("--text")
    search_cmd.add_argument("--image")
    search_cmd.add_argument("--top-k", type=int, default=10)
    search_cmd.add_argument("--category")
    search_cmd.add_argument("--style")
    search_cmd.add_argument("--use-case")
    search_cmd.add_argument("--generation-mode", choices=["text_to_3d", "image_to_3d"])
    search_cmd.set_defaults(func=search)

    stats_cmd = subparsers.add_parser("stats")
    stats_cmd.set_defaults(func=stats)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
