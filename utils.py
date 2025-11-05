import memory

config = {}

docs2text = lambda docs: "\n\n---\n\n\n".join(map(lambda x: x["content"], docs))

def sequence2text(sequence):
    res = "---\n\n"
    for i, step in enumerate(sequence):
        res += f"Step {i}:\n"
        for k, v in step.items():
            if k[0] != "_":
                res += f"{k}: {v}\n"
        res += "\n---\n\n"
    return res

evalStr = lambda s, var: eval(f"f'''{s}'''", None, var)

get_exec_import = lambda var: evalStr(config["EXEC_IMPORT"], var)

prices = {
    "gemini-embedding-001": 0.15/1e6,
    "gemini-2.5-pro": (1.25/1e6, 10/1e6),
    "gemini-2.5-flash": (0.3/1e6, 2.5/1e6)
}

def add_embedding_cost(res):
    total_cost = res.total_tokens * prices["gemini-embedding-001"]
    memory.session["cost"] += total_cost
    return total_cost

def add_generative_cost(res):
    v = res.model_version
    u = res.usage_metadata
    p = prices.get(v, (0, 0))
    input_token_count = u.prompt_token_count
    output_token_count = u.total_token_count - u.prompt_token_count
    total_cost = p[0] * input_token_count + p[1] * output_token_count
    memory.session["cost"] += total_cost
    if input_token_count > 2e5:
        print(f"[HAL] Warning: input token count({input_token_count}) exceeds 200k tokens. Cost estimation will be inaccurate.")
    return total_cost
