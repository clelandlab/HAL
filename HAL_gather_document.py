from google.genai import types
import memory
import json
from utils import add_generative_cost
from display import log

BASE_SYSTEM_INSTRUCTION = f"""You are a researcher preparing documents for a coming task. Your goal is to gather all relevant documents. You will be shown the main task, the documents gathered so far (each with an index), and a list of queries already searched.

Your task is to:
1.  **Filter:** Review all "Gathered Documents". Identify any that are irrelevant, redundant, or useless for the task. List their *indices* in the "remove" key.
2.  **Search:** Review the task and the *relevant* documents. Provide new search queries to find missing information or to recursively find documents/tools/methods mentioned.
3.  **Stop:** If you believe all sufficient documents have been gathered and no more searches are needed, provide an empty list for "search".
"""
RECURSIVE_SYSTEM_INSTRUCTION = '\n**You MUST recursively gather ALL documents/tools/methods refered by relevant search results INFINITELY.\n\nFor example, if a search result states "use X", "see X", "search X" or "refer to X"\nYou should put "X" in the search query list.\nLater, include all the searched documents in the output.**'

system_instruction = lambda recursive: BASE_SYSTEM_INSTRUCTION + (RECURSIVE_SYSTEM_INSTRUCTION if recursive else "")
get_user_content = lambda task, doc_section, query_section: f"""# Task:

{task}

# Gathered Documents:

{doc_section}

# Searched Queries:

{query_section}"""

filter_docs = lambda indices_to_remove, doc_id_list: [doc_id for index, doc_id in enumerate(doc_id_list) if index not in set(indices_to_remove)]

def gather_document(query, recursive=False):
    log("[HAL] Gathering documents...", "Gathering Documents")
    task = query
    doc_id_list = []
    searched_queries = []
    queries_to_search = [query]
    while queries_to_search:
        current_queries = list(queries_to_search)
        searched_queries.extend(current_queries)
        for q in current_queries:
            res = memory.search(q)
            for doc_id, _ in res:
                if doc_id not in doc_id_list:
                    doc_id_list.append(doc_id)
        queries_to_search.clear()
        gathered_docs = [memory.get(doc_id) for doc_id in doc_id_list]
        doc_texts = []
        for i, doc in enumerate(gathered_docs):
            doc_texts.append(f"Document {i}:\n\n{doc['content']}")
        doc_section = '\n\n---\n\n'.join(doc_texts)
        query_section = '\n'.join([f"- {q}" for q in searched_queries])
        user_content = get_user_content(task, doc_section, query_section)
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["remove", "search"], properties={
                "remove": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
                "search": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))}),
            system_instruction=system_instruction(recursive=recursive),
        )
        print(user_content)
        res = memory.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=config)
        add_generative_cost(res)
        res_json = json.loads(res.text)
        indices_to_remove = res_json.get("remove", [])
        new_queries = res_json.get("search", [])
        if indices_to_remove:
            doc_id_list = filter_docs(indices_to_remove, doc_id_list)
        if new_queries:
            queries_to_search.extend(new_queries)
            log(f"  - New queries: {new_queries}")
        else:
            break
    log(f"  > doc count: {len(doc_id_list)}")
    return [memory.get(doc_id) for doc_id in doc_id_list]
