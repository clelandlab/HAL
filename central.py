import time, os, json, gzip
import numpy as np
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
e_data = None # embedding data
m_data = None # meta data

cos_sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def init():
    global e_data, m_data
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, "embedding.gz"), 'rt') as f:
        try:
            e_data = json.load(f)
        except:
            e_data = {}
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, "meta.gz"), 'rt') as f:
        try:
            m_data = json.load(f)
        except:
            m_data = {}
init()

def add(doc_id, source, doc_path=None):
    score = 0
    doc_dir = os.path.join(config.CENTRAL_DATA_PATH, "documents")
    if doc_path:
        with open(doc_path, 'r') as f:
            doc = f.read()
    else:
        doc_path = os.path.join(doc_dir, f"{doc_id}.txt")
        doc = input("File contents: ")
        with open(doc_path, 'w') as f:
            f.write(doc)
    doc_embed = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=doc,
        config={"task_type": "retrieval_document"})
    m_entry = {"time": time.time(), "source": source, "score": score}
    e_entry = [e.values for e in doc_embed.embeddings]
    global e_data, m_data
    e_data[f"{doc_id}"] = e_entry[0]
    m_data[f"{doc_id}"] = m_entry
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, "embedding.gz"), 'wt') as f:
        json.dump(e_data, f, indent=4)
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, "meta.gz"), 'wt') as f:
        json.dump(m_data, f, indent=4)
    return
def delete(doc_id):
    global e_data, m_data
    doc_path = f"{config.CENTRAL_DATA_PATH}documents/{doc_id}.txt"
    os.remove(doc_path)
    del e_data[f"{doc_id}"]
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, "embedding.gz"), 'wt') as f:
        json.dump(e_data, f, indent=4)
    del m_data[f"{doc_id}"]
    with gzip.open(os.path.join(config.CENTRAL_DATA_PATH, "meta.gz"), 'wt') as f:
        json.dump(m_data, f, indent=4)
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
