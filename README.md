# HAL

## Config

Create a `config.json` with the following content:

```json
{
  "GEMINI_API_KEY": "your gemini API key",
  "MEMORY_DATA_PATH": "/path/to/the/memory/data.gz",
  "EXEC_IMPORT": "import time, os, sys, json, yaml, scipy\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
}
```

## API

```python
# initialization
HAL.init("Name", _config=None)

# main interface
HAL(query=None)

# properties
HAL.name = "HAL"
HAL.auto = 0

# session operations
HAL.reset()
HAL.save(path="session.json")
HAL.load(path="session.json")

# memory operations
HAL.search(query)
HAL.memorize(content, meta={ "source": HAL.name })

# low-level components
HAL.memory
HAL.display
# Agents
HAL.gather_document
HAL.sort
HAL.plan
HAL.answer
HAL.code
```
