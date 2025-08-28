import time, json, gzip, hashlib
import numpy as np
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
data = {}

# helper functions
cos_sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def load_data():
    global data
    with gzip.open(config.CENTRAL_DATA_PATH, 'rt') as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data
def save_data():
    with gzip.open(config.CENTRAL_DATA_PATH, 'wt') as f:
        json.dump(data, f)
def sha256str(s):
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()
def embed(content, task_type="retrieval_document"):
    try:
        return client.models.embed_content(model="gemini-embedding-exp-03-07", contents=content, config={"task_type": task_type}).embeddings[0].values
    except:
        return None

def init():
    load_data()
init()

# operations
def add(content, meta={}):
    doc_id = sha256str(content)
    data_dict = {"content": content, "embedding": embed(content)}
    meta["time"] = int(time.time())
    data_dict.update(meta)
    data[doc_id] = data_dict
    save_data()
    return doc_id
def get(doc_id):
    global data
    doc = dict(data[doc_id])
    doc.update({"id": doc_id})
    return doc
def delete(doc_id):
    del data[doc_id]
    save_data()
def search(q, n=5, threshold=0):
    scores = []
    score = 0
    q_embedding = embed(q, task_type="retrieval_query")
    for doc_id, data_dict in data.items():
        score = cos_sim(q_embedding, np.array(data_dict['embedding']))
        if score < threshold:
            continue
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True) # TODO: Optimize later
    res = [], score_res = []
    for entry in scores[:n]:
        res.append(get(entry[0]))
        score_res.append(entry[1])
    return res, score_res
