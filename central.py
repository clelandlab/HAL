import time, os, json, gzip, hashlib
import numpy as np
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
e_data = None # embedding data
m_data = None # meta data

# helper functions
cos_sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def load_gz(path):
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, path), 'rt') as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data
def save_gz(path, data):
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, path), 'wt') as f:
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
    global e_data, m_data
    e_data = load_gz("embedding.gz")
    m_data = load_gz("meta.gz")
init()

# operations
def add(content, meta={}):
    global e_data, m_data
    doc_id = sha256str(content)
    save_gz(f"documents/{doc_id}.gz", {"content": content})
    meta["time"] = int(time.time())
    e_data[doc_id] = embed(content)
    m_data[doc_id] = meta
    save_gz("embedding.gz", e_data)
    save_gz("meta.gz", m_data)
    return doc_id
def delete(doc_id):
    global e_data, m_data
    doc_path = os.path.join(config.CENTRAL_DATA_PATH, f"documents/{doc_id}.gz")
    os.remove(doc_path)
    del e_data[doc_id]
    del m_data[doc_id]
    save_gz("embedding.gz", e_data)
    save_gz("meta.gz", m_data)
    return
def search(q, n=5, threshold=0):
    scores = []
    global e_data, m_data
    score = 0
    q_embedding = embed(q, task_type="retrieval_query")
    for doc_id, embedding in e_data.items():
        score = cos_sim(q_embedding, np.array(embedding))
        if score < threshold:
            continue
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True) # TODO: Optimize later
    res = []
    for entry in scores[:n]:
        meta = m_data[entry[0]]
        content = load_gz(os.path.join(config.CENTRAL_DATA_PATH, f"documents/{entry[0]}.gz")).get("content")
        doc = {"content": content}
        doc.update(meta)
        res.append(doc)
    return res
