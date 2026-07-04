import os
import time
import json
import pickle
import hashlib
import logging
import importlib
import numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    redis_lib = importlib.import_module("redis")
    REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None
    REDIS_AVAILABLE = False

try:
    faiss = importlib.import_module("faiss")
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL        = int(os.getenv("REDIS_TTL", 3600))      # 1 hour cache
ARTIFACTS_DIR    = os.getenv("ARTIFACTS_DIR", "data")
TOP_K            = int(os.getenv("TOP_K", 10))
INTERACTION_THRESHOLD = int(os.getenv("INTERACTION_THRESHOLD", 3))

# ---------------------------------------------------------------------------
# Global State — loaded once at startup
# ---------------------------------------------------------------------------
STATE = {
    "als_model"         : None,
    "als_user_to_idx"   : None,
    "als_item_to_idx"   : None,
    "als_user_ids"      : None,
    "als_item_ids"      : None,
    "als_user_item"     : None,
    "item_factors_norm" : None,
    "user_factors_norm" : None,
    "faiss_index"       : None,
    "faiss_item_ids"    : None,
    "faiss_user_to_idx" : None,
    "lfm_model"         : None,
    "lfm_dataset"       : None,
    "lfm_item_features" : None,
    "lfm_user_map"      : None,
    "lfm_item_ids_list" : None,
    "n_lfm_items"       : 0,
    "purchases"         : {},
    "als_users"         : set(),
    "lfm_users"         : set(),
    "redis_client"      : None,
    "startup_time"      : None,
}

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts at startup. Clean up on shutdown."""
    logger.info("Loading recommendation artifacts...")
    t0 = time.time()

    try:
        # ALS artifacts
        with open(f"{ARTIFACTS_DIR}/als_artifacts.pkl", "rb") as f:
            als = pickle.load(f)
        STATE["als_model"]       = als["model"]
        STATE["als_user_to_idx"] = als["user_to_idx"]
        STATE["als_item_to_idx"] = als["item_to_idx"]
        STATE["als_user_ids"]    = als["user_ids"]
        STATE["als_item_ids"]    = als["item_ids"]
        STATE["als_user_item"]   = als["user_item_matrix"]
        logger.info(f"ALS loaded: {len(als['user_ids']):,} users, {len(als['item_ids']):,} items")

        # FAISS artifacts
        with open(f"{ARTIFACTS_DIR}/faiss_artifacts.pkl", "rb") as f:
            faiss_art = pickle.load(f)
        STATE["item_factors_norm"] = faiss_art["item_factors_norm"]
        STATE["user_factors_norm"] = faiss_art["user_factors_norm"]
        STATE["faiss_item_ids"]    = faiss_art["item_ids"]
        STATE["faiss_user_to_idx"] = faiss_art["user_to_idx"]

        if FAISS_AVAILABLE:
            try:
                STATE["faiss_index"] = faiss.read_index(
                    f"{ARTIFACTS_DIR}/faiss_index.bin"
                )
                logger.info(f"FAISS index loaded: {STATE['faiss_index'].ntotal:,} items")
            except Exception as e:
                logger.warning(f"FAISS index not found: {e} — using numpy fallback")

        # LightFM artifacts
        with open(f"{ARTIFACTS_DIR}/lightfm_artifacts.pkl", "rb") as f:
            lfm = pickle.load(f)
        STATE["lfm_model"]        = lfm["model_hybrid"]
        STATE["lfm_dataset"]      = lfm["dataset"]
        STATE["lfm_item_features"]= lfm["item_features_matrix"]
        user_map, _, item_map, _  = lfm["dataset"].mapping()
        STATE["lfm_user_map"]     = user_map
        STATE["lfm_item_ids_list"]= list(item_map.keys())
        STATE["n_lfm_items"]      = len(STATE["lfm_item_ids_list"])
        logger.info(f"LightFM loaded: {STATE['n_lfm_items']:,} items")

        # Routing split
        with open(f"{ARTIFACTS_DIR}/routing_split.pkl", "rb") as f:
            routing = pickle.load(f)
        STATE["als_users"] = routing["als_users"]
        STATE["lfm_users"] = routing["lightfm_users"]

        # Purchase history for exclusion
        import pandas as pd
        events = pd.read_csv(f"{ARTIFACTS_DIR}/../data/events.csv")
        STATE["purchases"] = (
            events[events["event"] == "transaction"]
            .groupby("visitorid")["itemid"]
            .apply(set).to_dict()
        )
        logger.info(f"Purchase history: {len(STATE['purchases']):,} users")

    except Exception as e:
        logger.error(f"Artifact loading failed: {e}")

    # Redis connection
    if REDIS_AVAILABLE:
        try:
            STATE["redis_client"] = redis_lib.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                decode_responses=True, socket_timeout=1
            )
            STATE["redis_client"].ping()
            logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Redis not available: {e} — caching disabled")
            STATE["redis_client"] = None

    STATE["startup_time"] = time.time() - t0
    logger.info(f"Startup complete in {STATE['startup_time']:.2f}s")

    yield  # App runs here

    logger.info("Shutting down...")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hybrid Recommender API",
    description="LightFM (new users) + ALS (returning users)",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    user_id: int
    top_k: Optional[int] = 10
    use_cache: Optional[bool] = True

class RecommendResponse(BaseModel):
    user_id: int
    engine: str
    recommendations: List[int]
    scores: List[float]
    cached: bool
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    als_loaded: bool
    lfm_loaded: bool
    redis_connected: bool
    startup_time_s: float

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _cache_key(user_id: int, top_k: int) -> str:
    return f"rec:{user_id}:{top_k}"

def _get_from_cache(key: str):
    if STATE["redis_client"] is None:
        return None
    try:
        val = STATE["redis_client"].get(key)
        return json.loads(val) if val else None
    except Exception:
        return None

def _set_cache(key: str, value: dict):
    if STATE["redis_client"] is None:
        return
    try:
        STATE["redis_client"].setex(key, REDIS_TTL, json.dumps(value))
    except Exception:
        pass

def _get_als_recs(user_id: int, top_k: int = 30):
    user_to_idx = STATE["faiss_user_to_idx"]
    if user_id not in user_to_idx:
        return [], []
    u_idx = user_to_idx[user_id]
    if u_idx >= len(STATE["user_factors_norm"]):
        return [], []
    u_vec = STATE["user_factors_norm"][u_idx:u_idx+1].astype(np.float32)

    if STATE["faiss_index"] is not None:
        scores, indices = STATE["faiss_index"].search(u_vec, top_k)
        items  = [int(STATE["faiss_item_ids"][i])
                  for i in indices[0] if i < len(STATE["faiss_item_ids"])]
        sc     = [float(s) for s in scores[0][:len(items)]]
    else:
        raw    = STATE["item_factors_norm"] @ u_vec.T
        idx    = np.argsort(raw.flatten())[::-1][:top_k]
        items  = [int(STATE["faiss_item_ids"][i])
                  for i in idx if i < len(STATE["faiss_item_ids"])]
        sc     = [float(raw.flatten()[i]) for i in idx[:len(items)]]
    return items, sc

def _get_lfm_recs(user_id: int, top_k: int = 30):
    if user_id not in STATE["lfm_user_map"]:
        return [], []
    u_idx  = STATE["lfm_user_map"][user_id]
    n      = STATE["n_lfm_items"]
    scores = STATE["lfm_model"].predict(
        user_ids=np.full(n, u_idx),
        item_ids=np.arange(n),
        item_features=STATE["lfm_item_features"]
    )
    top_idx = np.argsort(scores)[::-1][:top_k]
    items   = [int(STATE["lfm_item_ids_list"][i]) for i in top_idx]
    sc      = [float(scores[i]) for i in top_idx]
    return items, sc

def _exclude_purchased(items, scores, user_id):
    bought = STATE["purchases"].get(user_id, set())
    if not bought:
        return items, scores
    filtered = [(it, sc) for it, sc in zip(items, scores) if it not in bought]
    if not filtered:
        return items, scores
    its, scs = zip(*filtered)
    return list(its), list(scs)

def _normalize(scores):
    arr = np.array(scores, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return [0.5] * len(scores)
    return ((arr - lo) / (hi - lo)).tolist()

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        als_loaded=STATE["als_model"] is not None,
        lfm_loaded=STATE["lfm_model"] is not None,
        redis_connected=STATE["redis_client"] is not None,
        startup_time_s=STATE["startup_time"] or 0.0,
    )

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Main recommendation endpoint.
    Routes to ALS (returning users) or LightFM (new users).
    Applies purchase exclusion, score normalization.
    Uses Redis cache when available.
    """
    t0      = time.perf_counter()
    user_id = req.user_id
    top_k   = req.top_k or TOP_K

    # --- Check cache ---
    if req.use_cache:
        cache_key = _cache_key(user_id, top_k)
        cached    = _get_from_cache(cache_key)
        if cached:
            cached["cached"]     = True
            cached["latency_ms"] = (time.perf_counter() - t0) * 1000
            return RecommendResponse(**cached)

    # --- Routing decision ---
    n_interactions = 0
    u_to_i = STATE["faiss_user_to_idx"] or {}
    u_item = STATE["als_user_item"]
    if user_id in u_to_i:
        uidx = u_to_i[user_id]
        if u_item is not None and uidx < u_item.shape[0]:
            n_interactions = u_item[uidx].nnz

    use_als = (
        user_id in STATE["als_users"] and
        n_interactions >= INTERACTION_THRESHOLD
    )

    # --- Get raw recommendations ---
    if use_als:
        engine = "ALS"
        raw_items, raw_scores = _get_als_recs(user_id, top_k=top_k * 3)
    else:
        engine = "LightFM"
        raw_items, raw_scores = _get_lfm_recs(user_id, top_k=top_k * 3)

    if not raw_items:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations for user {user_id}"
        )

    # --- Post-processing ---
    raw_items, raw_scores = _exclude_purchased(raw_items, raw_scores, user_id)
    if raw_scores:
        raw_scores = _normalize(raw_scores)

    final_items  = raw_items[:top_k]
    final_scores = raw_scores[:top_k]

    latency_ms = (time.perf_counter() - t0) * 1000

    response_data = {
        "user_id"        : user_id,
        "engine"         : engine,
        "recommendations": final_items,
        "scores"         : final_scores,
        "cached"         : False,
        "latency_ms"     : latency_ms,
    }

    # --- Store in cache ---
    if req.use_cache:
        _set_cache(cache_key, response_data)

    return RecommendResponse(**response_data)

@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend_get(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    use_cache: bool = Query(default=True)
):
    """GET version of the recommend endpoint for easy browser testing."""
    return await recommend(
        RecommendRequest(user_id=user_id, top_k=top_k, use_cache=use_cache)
    )

@app.delete("/cache/{user_id}")
async def clear_user_cache(user_id: int):
    """Clear cached recommendations for a specific user."""
    if STATE["redis_client"] is None:
        return {"message": "Redis not available"}
    keys_deleted = 0
    for top_k in [5, 10, 20, 50]:
        key = _cache_key(user_id, top_k)
        keys_deleted += STATE["redis_client"].delete(key)
    return {"user_id": user_id, "keys_deleted": keys_deleted}

@app.get("/stats")
async def stats():
    """Return system statistics."""
    redis_info = {}
    if STATE["redis_client"]:
        try:
            info = STATE["redis_client"].info("stats")
            redis_info = {
                "hits"  : info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception:
            pass
    return {
        "als_users"      : len(STATE["als_users"]),
        "lfm_users"      : len(STATE["lfm_users"]),
        "faiss_items"    : len(STATE["faiss_item_ids"] or []),
        "lfm_items"      : STATE["n_lfm_items"],
        "redis_available": STATE["redis_client"] is not None,
        "redis_stats"    : redis_info,
        "startup_time_s" : STATE["startup_time"],
    }
