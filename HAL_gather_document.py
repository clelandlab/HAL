from google.genai import types
import memory
import json
from utils import docs2text, add_generative_cost
from display import log

system_instruction = f"""You are a researcher preparing documents for a coming task. Your goal is to gather all relevant documents from the database. You will be shown the main task, the documents gathered so far (each with an index), and a list of queries already searched.

Your task is to:
1. **Filter:** Review all gathered documents. Identify the documents that are irrelevant or useless for the task. List their *indices* in the "remove" key.
2. **Search:** Review the task and the *relevant* documents. Provide new search queries to find missing information or to recursively find documents/tools/methods mentioned. **Do NOT search for methods in common Python packages like "scipy", "numpy", etc. Do NOT search again for queries already in the "Searched Queries" list.**
3. **Stop:** If you believe all sufficient documents have been gathered and no more searches are needed, provide an empty list for "search".
"""

get_user_content = lambda task, docs, query_section: f"""# Task:

{task}

# Gathered Documents:

{docs2text(docs)}

# Searched Queries:

{query_section}"""

filter_docs = lambda indices_to_remove, doc_id_list: [doc_id for index, doc_id in enumerate(doc_id_list) if index not in set(indices_to_remove)]

def gather_document(query):
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
    while True:
        docs = map(memory.get, doc_ids)
        query_section = '\n'.join([f"- {q}" for q in searched_queries])
        user_content = get_user_content(query, docs, query_section)
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
            contents=user_content,
            config=config)
        add_generative_cost(res)
        res_json = json.loads(res.text)
        doc_ids = filter_docs(res_json.get("remove", []), doc_ids)
        new_queries = res_json.get("search", [])
        for q in new_queries:
            search(q)
            searched_queries.append(q)
        log(f"  - search: {new_queries} -> {len(doc_ids)}", "Gathering Documents")
        if len(new_queries) == 0:
            break
    log(f"  > doc count: {len(doc_ids)}")
    return [memory.get(doc_id) for doc_id in doc_ids]
