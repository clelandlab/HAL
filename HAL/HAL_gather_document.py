from google.genai import types
from . import memory
import json
from .utils import docs2text, add_generative_cost
from .display import log

system_instruction = f"""You are a researcher preparing documents for a coming task. Your goal is to gather all relevant documents from the database. You will be shown the main task, a list of queries already searched, and the documents gathered so far.

Your task is to:
1. **Filter:** Review all gathered documents. Identify the documents that are completely irrelevant or useless for the task. List their *indices* in the "remove" key.
2. **Stop:** If gathered documents are sufficient or relevant queries are already searched, provide an empty list for "search". You must provide search queries if all documents are removed in the previous step.
3. **Search:** Review the task and the *relevant* documents. Provide new search queries to find missing information or to recursively find documents/tools/methods mentioned in the relevant documents. **Do NOT search for methods in common Python packages like "scipy", "numpy", "matplotlib", "yaml", etc. Do NOT search queries that are already in the "Searched Queries" list.**
"""

user_content = lambda task, docs, query_section: f"""# Task:

{task}

# Searched Queries:

The following queries have already been searched and should NOT be in the "search" output:

{query_section}

# Gathered Documents:

{docs2text(docs)}"""

filter_docs = lambda indices_to_remove, doc_id_list: [doc_id for index, doc_id in enumerate(doc_id_list) if index not in set(indices_to_remove)]

def gather_document(query, max_iterations=6):
    log("[HAL] Gathering documents...", "Gathering Documents")
    doc_ids = []
    def search(keyword):
        res = memory.search(keyword)
        for doc_id, _ in res:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        return len(res)
    searched_queries = []
    search(query)
    for i in range(max_iterations):
        docs = map(memory.get, doc_ids)
        query_section = '\n'.join([f"- {q}" for q in searched_queries])
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["remove", "search"], properties={
                "remove": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
                "search": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))}),
            system_instruction=system_instruction
        )
        res = memory.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content(query, docs, query_section),
            config=config)
        add_generative_cost(res)
        res_json = json.loads(res.text)
        doc_ids = filter_docs(res_json.get("remove", []), doc_ids)
        new_queries = res_json.get("search", [])
        for q in new_queries:
            search(q)
            searched_queries.append(q)
        log(f"  {i}. search: {new_queries} -> {len(doc_ids)}", "Gathering Documents")
        if len(new_queries) == 0:
            break
    log(f"  > doc count: {len(doc_ids)}")
    return [memory.get(doc_id) for doc_id in doc_ids]
