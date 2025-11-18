from . import memory

config = {}

def docs2text(docs):
    res = ""
    for i, doc in enumerate(docs):
        res += f"--- Document {i} ---\n\n{doc['content']}\n\n"
    res += "--- End of Documents ---"
    return res

def sequence2text(sequence):
    res = ""
    for i, step in enumerate(sequence):
        res += f"--- Step {i} ---\n\n"
        for k, v in step.items():
            if k[0] != "_":
                res += f"{k}: {v}\n"
        res += "\n"
    res += "--- End of Steps ---"
    return res

evalStr = lambda s, var: eval(f"f'''{s}'''", None, var)

get_exec_import = lambda var: evalStr(config["EXEC_IMPORT"], var)

prices = {
    "gemini-embedding-001": 0.15/1e6,
    "gemini-2.5-pro": (1.25/1e6, 10/1e6),
    "gemini-2.5-flash": (0.3/1e6, 2.5/1e6),
    "gemini-3-pro-preview": (2/1e6, 12/1e6)
}

def add_embedding_cost(res):
    total_cost = res.total_tokens * prices["gemini-embedding-001"]
    memory.session["cost"] += total_cost
    return total_cost

def add_generative_cost(res):
    v = res.model_version
    u = res.usage_metadata
    if v not in prices:
        print(f"[HAL] Warning: model version {v} not found in prices list. Cost estimation will be inaccurate.")
    p = prices.get(v, (0, 0))
    input_token_count = u.prompt_token_count
    output_token_count = u.total_token_count - u.prompt_token_count
    total_cost = p[0] * input_token_count + p[1] * output_token_count
    memory.session["cost"] += total_cost
    if input_token_count > 2e5:
        print(f"[HAL] Warning: input token count({input_token_count}) exceeds 200k tokens. Cost estimation will be inaccurate.")
    return total_cost
