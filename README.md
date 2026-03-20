# HAL

Heuristic Autonomous Lab

> This tool is supposed to be run in Jupyter notebooks using JupyterLab. Examples are available in the `examples` folder.

## Config

Create a `config.json` with the following content:

```json
{
  "GEMINI_API_KEY": "your gemini API key",
  "MEMORY_DATA_PATH": "/path/to/the/memory/data.gz",
  "EXEC_IMPORT": "import time, os, sys, json, yaml, scipy\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
}
```

Or pass config dictionary to `init` function, check examples/mininum.ipynb

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

## Cite this work

https://arxiv.org/abs/2603.08801

```bibtex
@article{li2026large,
  title={Large Language Model-Assisted Superconducting Qubit Experiments},
  author={Li, Shiheng and Miller, Jacob M and Lee, Phoebe J and Andersson, Gustav and Conner, Christopher R and Joshi, Yash J and Karimi, Bayan and King, Amber M and Malc, Howard L and Mishra, Harsh and others},
  journal={arXiv preprint arXiv:2603.08801},
  year={2026}
}
```
