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

```python
HAL(query=None)

# config
HAL.name = "HAL"
HAL.auto = False
HAL.silent = False

# session operations
HAL.reset()
HAL.save(path="session.json")
HAL.load(path="session.json")

# memory operations
HAL.search(query)
HAL.memorize(content, meta={ "source": HAL.name })

# low-level memory operations
id = HAL.memory.add(content, meta={})
doc = HAL.memory.get(doc_id)
HAL.memory.delete(doc_id)
docs, scores = HAL.memory.search(q, maxn=5, cutoff_gradient=0.03, threshold=0.6):
HAL.memory.load()
HAL.memory.save()
```
