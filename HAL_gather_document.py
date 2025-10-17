from google.genai import types
import memory
from utils import client, add_generative_cost, docs2text

def gather_document(query, silent=False):
    ids = []
    def search(keyword: str) -> str:
        """search for the keyword in knowledge base.

        Args:
            keyword: search query

        Returns:
            Top search results. a string containing at most 5 documents ranked in decreasing relevance.
        """
        res = memory.search(keyword)
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
            formatted_list = [float(f"{score:.3f}") for id, score in res]
            print("  - search:", keyword, "->", formatted_list)
        return docs2text(new_docs)

    if not silent:
        print("[HAL] Gathering documents...")
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction='You are a researcher gathering documents for a task. Call search function to gather relevant documents for the task. Regardless the prompt of the user, ALWAYS ONLY output a string of comma-separated document numbers (no word, no space) that are relevant to the task. You are encouraged to call the search function multiple times to dig into complicated problems. Make sure you search all possible resources for the task.',
        tools=[search]
    )
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )
    add_generative_cost(res)
    try:
        index_list = list(map(int, res.text.split(',')))
    except:
        index_list = []
    if not silent:
        print("  > doc count:", len(index_list))
    return [memory.get(ids[i]) for i in index_list]

