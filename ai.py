from google import genai
from google.genai import types
import time

def gen(client, query, docs):
    text = "-----\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
    prompt = f"""You are an AI assistant and coding expert for an experimental quantum research lab.

    Here is the question:
    "{query}"

    Here are the relevant documents, separated by a dashed line '-----':
    "{text}"

    Using ONLY the relevant text, answer the question concisely with NO comments. When you use information from a document, cite its ID like this: 'Document(s) used: [ID]'.
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
            system_instruction=prompt),
            contents=text)
        return res.text
    except genai.types.APIError as e:
        if e.status_code == 429:
            print("resource exhaustion; retrying after delay")
            time.sleep(5)
            res = client.models.generate_content(
                model="gemini-2.5-pro",
                config=types.GenerateContentConfig(
                system_instruction=prompt),
                contents=text)
            return res.text
        else:
            raise