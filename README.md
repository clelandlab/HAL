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

## Get Started

```python
import HAL
HAL.init() # initialization: loading memory, setting up display, etc.
# HAL.auto = 3 # auto-execution: HAL will automatically execute up to 3 steps

HAL.reset() # this reset HAL session
HAL("Do something") # main interface: query HAL

HAL() # continue without user input
```

## API

```python
# initialization
HAL.init("Name", _config=None)

# main interface
HAL(query=None)

# properties
HAL.auto = 0
HAL.session = {}

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
