import memory

# embedding are not counted.
prices = {
    "gemini-2.5-pro": (1.25/1e6, 10/1e6),
    "gemini-2.5-flash": (0.3/1e6, 2.5/1e6),
}

def add_cost(res):
    v = res.model_version
    u = res.usage_metadata
    p = prices.get(v, (0, 0))
    input_token_count = u.prompt_token_count
    output_token_count = u.total_token_count - u.prompt_token_count
    total_cost = p[0] * input_token_count + p[1] * output_token_count
    memory.session["cost"] += total_cost
    return total_cost
