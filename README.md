# HAL

## Config

Create a `config.py` with the following content:

```python
GEMINI_API_KEY = "your gemini API key"

MEMORY_DATA_PATH = "/path/to/the/memory/data.gz"

EXEC_IMPORT = """import time, os, sys, json, yaml, scipy
import numpy as np
import matplotlib.pyplot as plt
"""
```

## API

### Memory

```python
id = memory.add(content, meta={})
doc = memory.get(doc_id)
memory.delete(doc_id)
docs, scores = memory.search(q, maxn=5, cutoff_gradient=0.03, threshold=0.6):
memory.load()
memory.save()
```
