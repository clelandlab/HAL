# skynet

A real skynet...

## Config

Create a `config.py` with the following content:

```python
GEMINI_API_KEY = "your gemini API key"

CENTRAL_DATA_PATH = "/path/to/the/central/data.gz"
```

## Central

```python
id = central.add(content, meta={})
doc = central.get(doc_id)
central.delete(doc_id)
docs, scores = central.search(q, n=5, threshold=0)
```
