import time, os, json, gzip, hashlib
import numpy as np
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
e_data = None # embedding data
m_data = None # meta data

cos_sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def load_gz(path):
    with gzip.open(path, 'rt') as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data
def save_gz(data, path):
    with gzip.open(path, 'wt') as f:
        json.dump(data, f)
def init():
    global e_data, m_data
    e_data = load_gz(os.path.join(config.CENTRAL_DATA_PATH, "embedding.gz"))
    m_data = load_gz(os.path.join(config.CENTRAL_DATA_PATH, "meta.gz"))
init()
def sha256str(s):
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()
def add(content, meta={}):
    doc_dir = os.path.join(config.CENTRAL_DATA_PATH, "documents")
    doc_id = sha256str(content)
    doc_path = os.path.join(doc_dir, f"{doc_id}.txt")
    with open(doc_path, 'w') as f:
        json.dump({"content": content}, f)
    doc_embed = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=content,
        config={"task_type": "retrieval_document"})
    m_entry = {"time": time.time(),
               "source": meta.get("source", "N/A"),
               "score": meta.get("score", 0)}
    e_entry = [e.values for e in doc_embed.embeddings]
    global e_data, m_data
    e_data[f"{doc_id}"] = e_entry[0]
    m_data[f"{doc_id}"] = m_entry
    save_gz(e_data, os.path.join(config.CENTRAL_DATA_PATH, "embedding.gz"))
    save_gz(m_data, os.path.join(config.CENTRAL_DATA_PATH, "meta.gz"))
    return doc_id
def delete(doc_id):
    global e_data, m_data
    doc_path = f"{config.CENTRAL_DATA_PATH}documents/{doc_id}.txt"
    os.remove(doc_path)
    del e_data[f"{doc_id}"]
    del m_data[f"{doc_id}"]
    save_gz(e_data, os.path.join(config.CENTRAL_DATA_PATH, "embedding.gz"))
    save_gz(m_data, os.path.join(config.CENTRAL_DATA_PATH, "meta.gz"))
    return
def search(q, n=5, threshold=0):
    result = []
    global e_data
    score = 0
    q_embed = client.models.embed_content(model="gemini-embedding-exp-03-07",
                                        contents=q,
                                        config={"task_type": "retrieval_query"})
    q_embedding = np.array(q_embed.embeddings[0].values)
    for doc_id, embedding in e_data.items():
        score = cos_sim(q_embedding, np.array(embedding))
        if score >= threshold:
            result.append((doc_id, score))
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:n]