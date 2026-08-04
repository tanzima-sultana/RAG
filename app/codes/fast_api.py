
from fastapi import FastAPI
import os 

from . import state
from .query import router as query_router
from constants import LOCAL, AWS, FIXED, SENTENCE, SEMANTIC, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

app = FastAPI()

@app.on_event("startup")
def startup_event():
    # Read params
    mode = os.environ["mode"]
    model_name = os.environ["model_name"]
    dataset_size = int(os.environ["dataset_size"])
    chunking_type = os.environ["chunking_type"]
    index_type = os.environ["index_type"]

    state.load_state(mode, model_name, dataset_size, chunking_type, index_type)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(query_router)