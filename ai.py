from google import genai
from google.genai import types
import time, json
import memory
from cost import add_generative_cost, add_embedding_cost

docs2text = lambda docs: "\n\n-----\n\n\n".join(map(lambda x: x["content"], docs))

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
    res = memory.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )
    add_generative_cost(res)
    return list(docs.values())

def question(query, docs=[], silent=False):
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
    add_generative_cost(res)
    return res.text

def code(query, docs=[], exec_import="", silent=False):
    text = docs2text(docs)
    system_instruction = f"You are a world class programming AI that generates Python code based on user requirements. Write the code using ONLY the following documents:\n\n\n{text}\n\n\nThe code should be self-contained and runnable. Do NOT include any side behaviors like printing messages. Absolutely do NOT include any comments or explanations. The following packages are already imported (do NOT import them again!):\n{exec_import}"
    if not silent:
        print("[HAL] Coding...")
    res = memory.client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["code"], properties={ "code": types.Schema(type=genai.types.Type.STRING) }),
            system_instruction=system_instruction
        ),
        contents=query
    )
    add_generative_cost(res)
    return json.loads(res.text)["code"]
