import re
from google.genai import types
import memory
import json
from utils import add_generative_cost, docs2text
from display import log

def extract_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)

def filter_docs(gathered_docs, indices_to_remove, doc_ids_seen):
    if not indices_to_remove:
        return gathered_docs, doc_ids_seen
    indices_to_remove_set = set(indices_to_remove)
    new_gathered_docs = []
    removed_ids = []
    for i, doc in enumerate(gathered_docs):
        if i in indices_to_remove_set:
            removed_ids.append(doc['id'])
        else:
            new_gathered_docs.append(doc)
    for doc_id in removed_ids:
        if doc_id in doc_ids_seen:
            doc_ids_seen.remove(doc_id)
    return new_gathered_docs, doc_ids_seen

def gather_document(query, recursive=False):
    log("[HAL] Gathering documents...", "Gathering Documents")
    system_instruction = """You are a researcher preparing documents for a coming task.
    Your goal is to gather all relevant documents.
    You will be shown the main task, the documents gathered so far (each with an index), and a list of queries already searched.

    Your task is to:
    1.  **Filter:** Review all "Gathered Documents". Identify any that are irrelevant, redundant, or useless for the task. List their *indices* in the "remove" key.
    2.  **Expand:** Review the task and the *relevant* documents. Provide new search queries to find missing information or to recursively find documents/tools/methods mentioned.
    3.  **Stop:** If you believe all sufficient documents have been gathered and no more searches are needed, provide an empty list for "search".

    You MUST output a valid JSON object in the following format:
    {
    "remove": [0, 3],
    "search": ["new query 1", "query about X"]
    }
    """
    if recursive:
        system_instruction += '**You MUST recursively gather ALL documents/tools/methods refered by relevant search results INFINITELY. You are strongly encouraged to call the search function multiple times.\n\nFor example, if a search result states "use X", "see X", "search X" or "refer to X"\nYour action should be search("X")\nLater, include all the searched documents in the output.**'
    task = query
    gathered_docs = []
    doc_ids_seen = set()
    searched_queries = set()
    queries_to_search = [query]
    iteration = 0
    while queries_to_search:
        iteration += 1
        log(f"  --- Iteration {iteration} ---")
        current_queries = list(queries_to_search)
        queries_to_search.clear()
        for q in current_queries:
            if q in searched_queries:
                continue
            searched_queries.add(q)
            try:
                res = memory.search(q)
                log(f"  - search: {q} -> {len(res)} results")
            except Exception as e:
                log(f"  ! Error: {e}")
                continue
            for doc_id, score in res:
                if doc_id in doc_ids_seen:
                    continue
                doc_ids_seen.add(doc_id)
                doc = memory.get(doc_id)
                gathered_docs.append(doc)
        doc_texts = []
        if not gathered_docs:
            doc_texts.append("No documents gathered yet.")
        else:
            for i, doc in enumerate(gathered_docs):
                doc_texts.append(f"Document {i}:\n{doc['content']}")
        query_texts = [f"- {q}" for q in searched_queries]
        doc_section = '\n\n---\n\n'.join(doc_texts)
        query_section = '\n'.join(query_texts)
        user_content = f"""# Task:
        {task}

        # Gathered Documents:
        {doc_section}

        # Searched Queries:
        {query_section}
        """
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )
        try:
            res = memory.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_content,
                config=config
            )
            add_generative_cost(res)
        except Exception as e:
            log(f"  ! Error: {e}")
            break
        try:
            res_json = extract_json(res.text)
            indices_to_remove = res_json.get("remove", [])
            new_queries = res_json.get("search", [])
            if indices_to_remove:
                log(f"  - Removed docs: {indices_to_remove}")
                gathered_docs, doc_ids_seen = filter_docs(gathered_docs, indices_to_remove, doc_ids_seen)
            if new_queries:
                for nq in new_queries:
                    if nq not in searched_queries:
                        queries_to_search.append(nq)
                if queries_to_search:
                    log(f"  - New queries: {queries_to_search}")
                else:
                    log(f"  - Only duplicate queries provided.")
            if not queries_to_search:
                log("  > Sufficient documents/no new queries. Stopping.")
                break
        except Exception as e:
            log(f"  ! Error: {e}. Output: {res.text}")
            break
    log(f"  > doc count: {len(gathered_docs)}")
    return gathered_docs