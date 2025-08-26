from google import genai
from google.genai import types
import time
from central import search

def gather_document(client, query, docs=[]):
    search_declaration = {
        "name": "ai_search",
        "description": "Searches for additional information to answer the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string"
                },
            },
            "required": ["keyword"]
        }
    }
    tools = types.Tool(function_declarations=[search_declaration])
    def ai_search(keyword):
        print("ai_search called")
        docs_added = []
        new_docs = search(keyword)
        for d in new_docs:
            if d in docs:
                print("Document already presented: ", d['id'])
            else:
                docs.append(d)
                docs_added.append(d['id'])
            res = f"Added docs: {', '.join(docs_added)}" if docs_added else "no new docs added"
            return {"Result": res}
    while True:
        text = "-----\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
        system_instruction = """Given the following text documents, determine if there is enough information to solve the problem.
        If more information is needed, you MUST call the ai_search function with your recommended keyword.
        """
        config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    tools=[tools]
                )
        contents = [types.Content(role="user", parts=[types.Part(text=f"Problem: {query}\n\nDocuments:\n{text}")])]
        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                config=config,
                contents=contents)
        except genai.types.APIError as e:
            if e.status_code == 429:
                print("resource exhaustion; retrying after delay")
                time.sleep(5)
                res = client.models.generate_content(
                model="gemini-2.5-flash",
                config=config,
                contents=contents)
            else:
                raise
        tool_call = res.candidates[0].content.parts[0].function_call
        print("tool call: ", tool_call)
        if not tool_call:
            print("complete search")
            break
        result = ai_search(**tool_call.args)
        f_res_part = types.Part.from_function_response(name=tool_call.name, response={"result": result})
        contents.append(res.candidates[0].content)
        contents.append(types.Content(role="user", parts=[f_res_part]))
    return res.text

def gen(client, query, docs):
    text = "\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
    system_instruction = "You are an AI assistant and coding expert for an experimental quantum research lab. Answer the question concisely with NO comments and using ONLY the provided relevant text."
    contents = f"""Here is the question:
    "{query}"
    Here are the relevant documents, separated by a dashed line '-----':
    "{text}"
    """
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