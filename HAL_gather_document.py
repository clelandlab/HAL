from google.genai import types
import memory
import json
from utils import add_generative_cost
from display import log

BASE_SYSTEM_INSTRUCTION = f"""You are a researcher preparing documents for a coming task. Your goal is to gather all relevant documents. You will be shown the main task, the documents gathered so far (each with an index), and a list of queries already searched.

Your task is to:
1.  **Filter:** Review all "Gathered Documents". Identify any that are irrelevant, redundant, or useless for the task. List their *indices* in the "remove" key.
2.  **Expand:** Review the task and the *relevant* documents. Provide new search queries to find missing information or to recursively find documents/tools/methods mentioned.
3.  **Stop:** If you believe all sufficient documents have been gathered and no more searches are needed, provide an empty list for "search".

You MUST output a valid JSON object in the following format:
{
"remove": [0, 3],
"search": ["new query 1", "query about X"]
}"""
RECURSIVE_SYSTEM_INSTRUCTION = '\n**You MUST recursively gather ALL documents/tools/methods refered by relevant search results INFINITELY. You are strongly encouraged to call the search function multiple times.\n\nFor example, if a search result states "use X", "see X", "search X" or "refer to X"\nYour action should be search("X")\nLater, include all the searched documents in the output.**'

system_instruction = lambda recursive: BASE_SYSTEM_INSTRUCTION + (RECURSIVE_SYSTEM_INSTRUCTION if recursive else "")
user_content = lambda task, doc_section, query_section: f"""# Task:
{task}

# Gathered Documents:
{doc_section}

# Searched Queries:
{query_section}"""

def extract_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)

def filter_docs(indices_to_remove, doc_id_list):
    return [doc_id for index, doc_id in enumerate(doc_id_list) if index not in set(indices_to_remove)]

def gather_document(query, recursive=False):
    log("[HAL] Gathering documents...", "Gathering Documents")
    task = query
    doc_id_list = []
    searched_queries = []
    queries_to_search = [query]
    iteration = 0
    while queries_to_search:
        iteration += 1
        log(f"  --- Iteration {iteration} ---")
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
        if not gathered_docs:
            doc_texts.append("No documents gathered yet.")
        else:
            for i, doc in enumerate(gathered_docs):
                doc_texts.append(f"Document {i}:\n{doc['content']}")
        doc_section = '\n\n---\n\n'.join(doc_texts)
        query_section = '\n'.join([f"- {q}" for q in searched_queries])
        user_content = user_content(task, doc_section, query_section)
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["remove", "search"], properties={
                "remove": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
                "search": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))}),
            system_instruction=system_instruction(recursive=recursive),
        )
        res = memory.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=config)
        add_generative_cost(res)
        res_json = extract_json(res.text)
        indices_to_remove = res_json.get("remove", [])
        new_queries = res_json.get("search", [])
        if indices_to_remove:
            doc_id_list = filter_docs(indices_to_remove, doc_id_list)
        if new_queries:
            queries_to_search.extend(new_queries)
            log(f"  - New queries: {new_queries}")
        else:
            log("  > Sufficient documents/no new queries. Stopping.")
            break
    log(f"  > doc count: {len(doc_id_list)}")
    return [memory.get(doc_id) for doc_id in doc_id_list]