from google import genai
from google.genai import types
import time
import central

docs2text = lambda docs: "\n\n-----\n\n\n".join(map(lambda x: x["content"], docs))

def gather_document(query):
    docs = {}
    def search(keyword: str) -> str:
        """search for the keyword in knowledge base.

        Args:
            keyword: search query

        Returns:
            search results. a string containing multiple documents ranked in decreasing relevance.
        """
        print("search:", keyword)
        new_docs = central.search(keyword, n=3, threshold=0.6)
        for d in new_docs:
            if d["id"] in docs:
                d["content"] = "Document already presented."
                continue
            docs[d["id"]] = d
        return docs2text(new_docs)

    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction='You are a researcher gathering documents for a task. Call search function to gather information for the task. Do NOT solve or complete the task. Regardless the prompt of the user, ALWAYS ONLY output text "complete" if you think the searched documents are sufficient to complete the task. You may call the search function multiple times to dig into complicated problems',
        tools=[search]
    )
    res = central.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )
    return list(docs.values())

def gen(query, docs):
    text = "\n".join(f"Document ID: {d['id']}: \n {d['content']}" for d in docs)
    system_instruction = "You are an AI assistant and coding expert for an experimental quantum research lab. Answer the question concisely with NO comments and using ONLY the provided relevant text."
    contents = f"""Here is the question:
    "{query}"
    Here are the relevant documents, separated by a dashed line '-----':
    "{text}"
    """

    try:
        res = central.client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
            system_instruction=system_instruction),
            contents=contents)
        return res.text
    except genai.types.APIError as e:
        if e.status_code == 429:
            print("resource exhaustion; retrying after delay")
            time.sleep(5)
            res = central.client.models.generate_content(
                model="gemini-2.5-pro",
                config=types.GenerateContentConfig(
                system_instruction=system_instruction),
                contents=contents)
            return res.text
        else:
            raise

def start_chat():
    while True:
        user_input = input("You: ")
        if user_input == 'q' or user_input == 'quit': break
        docs = gather_document(user_input)
        response = gen(user_input, docs)
        print(f"Gemini: {response.text}")