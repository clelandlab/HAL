from google import genai
from google.genai import types
import time

def check(client, query, docs):
    text = "-----\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
    system_instruction = """Given the following text documents, determine if there is enough information to solve the problem.
    If there is, set 'sufficient' to true.
    If not, set 'sufficient' to false, and recommend a specific keyword or query to search for the missing information in the 'search_query' field.
    The output must be a JSON object with the following structure:
    {
        "sufficient": true | false
        "search_query": "specific keyword or query to search"
    }
    """
    contents = f"Problem: {query}\n\nDocuments:\n{text}"
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_budget=0)), # Turn off thinking
            contents=contents)
        return res.text
    except genai.types.APIError as e:
        if e.status_code == 429:
            print("resource exhaustion; retrying after delay")
            time.sleep(5)
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                system_instruction=system_instruction),
                contents=contents)
            return res.text
        else:
            raise

def gen(client, query, docs):
    text = "\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
    system_instruction = "You are an AI assistant and coding expert for an experimental quantum research lab. When you use information from a document, cite its ID like this: 'Document(s) used: [ID]'. Answer the question concisely with NO comments and using ONLY the provided relevant text."
    contents = f"Here is the question:\n\"{query}\"\n\nHere are the relevant documents, separated by a dashed line '-----':\n\"{text}\""
    try:
        res = client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
            system_instruction=system_instruction),
            contents=contents)
        return res.text
    except genai.types.APIError as e:
        if e.status_code == 429:
            print("resource exhaustion; retrying after delay")
            time.sleep(5)
            res = client.models.generate_content(
                model="gemini-2.5-pro",
                config=types.GenerateContentConfig(
                system_instruction=system_instruction),
                contents=contents)
            return res.text
        else:
            raise