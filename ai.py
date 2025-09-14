from google import genai
from google.genai import types
import time
import memory

docs2text = lambda docs: "\n\n-----\n\n\n".join(map(lambda x: x["content"], docs))

def gather_document(query, silent=False):
    docs = {}
    def search(keyword: str) -> str:
        """search for the keyword in knowledge base.

        Args:
            keyword: search query

        Returns:
            Top 3 search results. a string containing at most three documents ranked in decreasing relevance.
        """
        new_docs, scores = memory.search(keyword, n=3, threshold=0.6)
        for d in new_docs:
            if d["id"] in docs:
                d["content"] = "Document already presented."
                continue
            docs[d["id"]] = d
        if not silent:
            formatted_list = [f"{item:.3f}" for item in scores]
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
    res = memory.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )
    return list(docs.values())

def question(query, silent=False):
    docs = gather_document(query, silent=silent)
    text = docs2text(docs)
    system_instruction = f"You are a researcher on experimental quantum computing. Answer the question concisely with NO comments and using ONLY the following documents:\n\n\n{text}"
    if not silent:
        print("[HAL] Thinking...")
    res = memory.client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction
        ),
        contents=query
    )
    return res.text

