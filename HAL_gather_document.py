from google import genai
from google.genai import types
import memory
from utils import client, add_generative_cost, docs2text

def gather_document(query, silent=False):
    docs = {}
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
            if d["id"] in docs:
                d["content"] = "Document already presented."
                continue
            docs[d["id"]] = d
            new_docs.append(d)
        if not silent:
            formatted_list = [float(f"{score:.3f}") for id, score in res]
            print("  > search:", keyword, "->", formatted_list)
        return docs2text(new_docs)

    if not silent:
        print("[HAL] Gathering documents...")
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction='You are a researcher gathering documents for a task. Call search function to gather information for the task. Do NOT solve or complete the task. Regardless the prompt of the user, ALWAYS ONLY output text "complete" if you think the searched documents are sufficient to complete the task. You are encouraged to call the search function multiple times to dig into complicated problems',
        tools=[search]
    )
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )
    add_generative_cost(res)
    return list(docs.values())


