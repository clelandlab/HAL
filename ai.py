from google import genai
from google.genai import types
from google.genai.types import ModelContent, Part, UserContent
import time

def gen(client, query, docs, history):
    text = "\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
    prompt = f"""You are an AI assistant and coding expert for an experimental quantum research lab.

    Here is the question:
    "{query}"

    Here is the relevant information:
    "{text}"

    Here is the history:
    "{history}

    Using ONLY the relevant information and history, answer the question concisely with NO comments. When you use information from a document, cite its ID like this: 'Document(s) used: [ID]'.
    """
    history.append(UserContent(parts=[Part(text=query)]))
    try:
        res = client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
            system_instruction=prompt),
            contents=history)
        history.append(ModelContent(parts=[Part(text=res.text)]))
        return res.text, history
    except genai.types.APIError as e:
        if e.status_code == 429:
            print("resource exhaustion; retrying after delay")
            time.sleep(5)
            res = client.models.generate_content(
                model="gemini-2.5-pro",
                config=types.GenerateContentConfig(
                system_instruction=prompt),
                contents=history)
            history.append(ModelContent(parts=[Part(text=res.text)]))
            return res.text, history
        else:
            raise