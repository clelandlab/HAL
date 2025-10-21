import re
from google.genai import types
import memory
from utils import client, add_generative_cost, docs2text

def gather_document(query, recursive=False, silent=False):
    ids = []
    def search(query: str) -> str:
        """search in knowledge base.

        Args:
            query: one search query at a time.

        Returns:
            Top search results, containing at most 5 documents ranked in decreasing relevance.
        """
        res = memory.search(query)
        new_docs = []
        for id, score in res:
            d = memory.get(id)
            new_docs.append(d)
            if d["id"] in ids:
                d["content"] = "Document already presented."
                continue
            d["content"] = f"Document {len(ids)}: \n\n" + d["content"]
            ids.append(d["id"])
        if not silent:
            score_list = [float(f"{score:.3f}") for id, score in res]
            index_list = [ids.index(id) for id, score in res]
            print("  - search:", query, "->", index_list, score_list)
        return docs2text(new_docs)

    if not silent:
        print("[HAL] Gathering documents...")
    system_instruction = "You are a researcher gathering documents for a task. Call search function to gather relevant documents for the task. You are encouraged to call the search function multiple times to dig into complicated problems. Make sure you search all possible documents for the task.\n"
    if recursive:
        system_instruction += 'You MUST recursively search for more documents refered by previously gathered relevant documents. For example, search for X if a relevant document says something like "see X" or "search X" or "refer to X".\n'
    system_instruction += "\nRegardless the prompt of the user, ALWAYS ONLY output comma-separated document numbers that are relevant to the task. Output format example: 0,2,3,5"
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_instruction,
        tools=[search]
    )
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )
    add_generative_cost(res)
    try:
        text = re.sub(r'[^0-9,]', '', res.text)
        index_list = list(map(int, text.split(',')))
    except:
        index_list = []
        print("  ! Error. Model output:", res.text)
    if not silent:
        print(f"  > doc count: {len(index_list)} [{text}]")
    return [memory.get(ids[i]) for i in index_list if i < len(ids)]

