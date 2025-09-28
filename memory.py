import time, json, gzip, hashlib
import numpy as np
from google import genai
import config
from cost import add_cost

session = {
    "cost": 0.0
}

client = genai.Client(api_key=config.GEMINI_API_KEY)
data = {}

# helper functions
cos_sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def load():
    global data
    with gzip.open(config.MEMORY_DATA_PATH, 'rt') as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data
def save():
    with gzip.open(config.MEMORY_DATA_PATH, 'wt') as f:
        json.dump(data, f)
def sha256str(s):
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()
def embed(content, task_type="retrieval_document"):
    try:
        return client.models.embed_content(model="gemini-embedding-001", contents=content, config={"task_type": task_type}).embeddings[0].values
    except:
        return None

def init():
    load()
init()

# operations
def add(content, meta={}):
    doc_id = sha256str(content)
    data_dict = {"content": content, "embedding": embed(content)}
    meta["time"] = int(time.time())
    data_dict.update(meta)
    data[doc_id] = data_dict
    return doc_id
def get(doc_id):
    global data
    doc = dict(data[doc_id])
    doc.update({"id": doc_id})
    return doc
def delete(doc_id):
    del data[doc_id]

# return a list of (doc_id, score)
def search(q, maxn=5, cutoff_gradient=0.03, threshold=0.6):
    scores = []
    score = 0
    q_embedding = embed(q, task_type="retrieval_query")
    for doc_id, data_dict in data.items():
        score = cos_sim(q_embedding, np.array(data_dict['embedding']))
        if score < threshold:
            continue
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    for i in range(maxn):
        if scores[i][1] - scores[i+1][1] >= cutoff_gradient:
            return scores[:(i+1)]
    return scores[:maxn]
