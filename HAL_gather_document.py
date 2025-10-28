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
            Top search results ranked in decreasing relevance.
        """
        res = memory.search(query)
        new_docs = []
        for id, score in res:
            d = memory.get(id)
            new_docs.append(d)
            if d["id"] in ids:
                d["content"] = f"Document {ids.index(d['id'])}: already presented."
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
    system_instruction = "You are a research librarian preparing documents for a coming task. You MUST call the search function to gather relevant documents for the task. You can call the search function multiple times to gather all relevant documents for the task. Always search for documents instead of assuming.\n\n**When you gathered sufficient documents, output a list of numbers indicating the indices of relevant document. Do NOT attempt to solve the problem!**\noutput format example: 0,2,3,6\n\n"
    if recursive:
        system_instruction += '**You MUST recursively gather ALL documents/tools/methods refered by relevant search results INFINITELY. You are strongly encouraged to call the search function multiple times.\n\nFor example, if a search result states "use XXX", "see XXX", "search XXX" or "refer to XXX"\nYour action should be search("XXX")\nLater, include all the searched documents in the output.**'
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
    text = ""
    try:
        text = re.sub(r'[^0-9,]', '', res.text)
        index_list = list(map(int, text.split(',')))
    except:
        index_list = range(len(ids))
        print(f"  ! Error: {res.finish_reason}. Model output: {res.text}")
    if not silent:
        print(f"  > doc count: {len(index_list)} [{text}]")
    return [memory.get(ids[i]) for i in index_list if i < len(ids)]

