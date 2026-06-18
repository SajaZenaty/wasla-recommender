
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 32


FAISS_TOP_K = 40
MIN_CANDIDATES = 20

ACTION_WEIGHTS = {
    "click": 1, 
    "save": 2, 
    "apply": 4
}
LAMBDA_DECAY = 0.05

SCORING_WEIGHTS = {
    "semantic": 0.35,
    "retrieval": 0.15,
    "cf": 0.15,
    "category": 0.10,
    "location": 0.08,
    "time_fit": 0.07,
    "freshness": 0.05,
    "trust": 0.05
}

TIME_BALANCE_THRESHOLDS = {
    "low": 5,
    "high": 20,
    "bonus": 0.15,
    "penalty": -0.05
}

