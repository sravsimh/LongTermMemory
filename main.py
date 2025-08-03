import os
import json
import uuid
import requests
from openai import OpenAI
from transformer import createEmbeddings
from vector_embeddings import createQdrant, addToQdrant, deleteQdrant, searchQdrant

# --- Configuration ---
# IMPORTANT: Set up your environment variables before running.
# 1. Get your Gemini API key from Google AI Studio.
# 2. Set the GEMINI_API_KEY environment variable.

os.environ["GEMINI_API_KEY"] = "your-api-key"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

# Or use os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "your-api-key"
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Helper function to call Gemini API ---


def call_openai_api(prompt, is_json=False):
    """Sends a prompt to OpenAI's API and returns the response."""
    try:
        kwargs = {
            "model": "gpt-4o-mini",
            "input": prompt
        }

        # Only include text format if JSON is requested
        if is_json:
            kwargs["text"] = {"format": {"type": "json_object"}}

        response = client.responses.create(**kwargs)

        return response.output[0].content[0].text

    except Exception as e:
        return f"OpenAI API error: {e}"


def call_gemini_api(prompt, is_json=False):
    """Sends a prompt to the Gemini API and returns the response."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set."

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"} if is_json else {}
    }

    try:
        response = requests.post(
            API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()

        if result.get("candidates"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Sorry, I received an unexpected response from the API."

    except requests.exceptions.RequestException as e:
        print(f"Gemini API call error: {e}")
        return "Sorry, I'm having trouble connecting to my brain right now."
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {e}")
        return "Sorry, I couldn't understand the response from my brain."

# --- Core Logic: Handling user messages ---


def handle_user_message(user_id, user_message_text):
    """
    Processes a user's message, manages memories, and generates a response.
    """

    # 1. Deletion Check
    deletion_prompt = f"""
    You are an intelligent assistant helping manage long-term user memories.

    Analyze the following user message to determine if it expresses a clear intent to delete, stop using, replace, or update a previously stored memory.

    Message: "{user_message_text}"

    Your task is to:
    1. Detect whether the user wants to forget or update any specific information.
    2. Extract the specific piece(s) of information that should be removed from memory.

    Respond in strict JSON format with:
    - "shouldDelete": boolean (true if the user intends to remove or update memory)
    - "memoryToForget": a list of strings, each describing the specific memory content to be forgotten (or an empty list if none)

    Examples:
    - Input: "I don't use Brave anymore"
    Output: {{ "shouldDelete": true, "memoryToForget": ["Brave"] }}

    - Input: "Replace Jira with Linear for project tracking"
    Output: {{ "shouldDelete": true, "memoryToForget": [
        "Jira for project tracking"] }}

    - Input: "I still use Notion every day"
    Output: {{ "shouldDelete": false, "memoryToForget": [] }}

    Only return the JSON. Do not include any explanation or additional text.
    """

    deletion_response_text = call_openai_api(deletion_prompt, is_json=True)
    try:
        deletion_response = json.loads(deletion_response_text)
        if deletion_response.get("shouldDelete") and isinstance(deletion_response.get("memoryToForget"), list):
            subject_to_forget = deletion_response["memoryToForget"]
            vectors = createEmbeddings(subject_to_forget)
            for i, vec in enumerate(vectors):
                searchResults = searchQdrant(user_id, vec, True)
                if (len(searchResults) > 0):
                    for res in searchResults:
                        delValue = deleteQdrant(user_id, res.id)
                        if delValue:
                            print(
                                f"I have forgotten about {res.payload['content']}")
                        else:
                            return f"error deleting Qdrant Vector"

                else:

                    return f"I don't have a memory about \"{subject_to_forget[i]}\"."
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Could not parse deletion response: {e}")

    # 2. Memory Creation Check
    creation_prompt = f"""
        Analyze the following user message to identify if it contains a core, long-term fact about the user and make sure that the intent is not to delete a memory or talking about an exsisting memory ex: i no loger use __ or i do not use __. where user is trying to express you dislike or negative emotion about a certain entity.
        Do not store trivial or conversational filler.
        Message: "{user_message_text}"
        If a core fact is present, summarize it. Respond in JSON with "shouldRemember" (boolean) and "memory" (an array of strings for each fact, or null).
        Example: For "I use Shram and Magnet for my work", return {{"shouldRemember": "True", "memory": ["uses Shram for work", "uses Magnet for work"]}}.
        Make sure not to mix multiple things into one memory listItem; each should contain only one entity.
    """
    creation_response_text = call_openai_api(creation_prompt, is_json=True)
    try:
        creation_response = json.loads(creation_response_text)
        if creation_response.get("shouldRemember") and isinstance(creation_response.get("memory"), list):
            try:
                new_memories = creation_response["memory"]
                vec_new_mem = createEmbeddings(new_memories)
                embeds = {}
                embeds["payload"] = []
                embeds["vector"] = vec_new_mem
                for i, vec in enumerate(vec_new_mem):
                    embeds["payload"].append(
                        {"status": "True", "content": new_memories[i]})
                adding = addToQdrant(user_id, embeds)
                if adding == None:
                    exit(1)
                print(f"added {new_memories} to memory")
            except Exception as e:
                return (f"Error creating Memories:", e)

    except (json.JSONDecodeError, TypeError) as e:
        print(f"Could not parse creation response: {e}")

    casual_prompt_text = f'''
    Analyse the use prompt and tell if its a casual promt or a prompt which requires knowledge or the data regarding the user,
    message: {user_message_text} and return output in json format like {{"requiresMemory":True}} else {{"requiresMemory":False}}  and make sure the T and F are caps as im using python
    '''
    casual_prompt_text_response = call_openai_api(
        casual_prompt_text, is_json=True)

    try:

        casual_response = json.loads(casual_prompt_text_response)

        if casual_response.get("requiresMemory"):

            # 3. Generate a response using context and memories
            res = searchQdrant(user_id, createEmbeddings(
                user_message_text)[0])
            indexedMatches = []
            if len(res) > 0:
                for i in res:
                    indexedMatches.append(i.payload["content"])
                memories_context = "Here are some things I know about the user:\n- " + \
                    "\n- ".join(indexedMatches)
            else:
                memories_context = "I don't have any memories about the user yet."

            response_prompt = f"""
                You are a helpful assistant with a long-term memory with add to memory and delete from memory capabilities. the below is your current memory fetch with Semantic search of the user message.
                {memories_context}
                Based on this context and our current conversation, provide a helpful and relevant response to the user's latest message.
                User's message: "{user_message_text}"
            """
            final_response = call_openai_api(response_prompt)

            return final_response
        else:
            final_response = call_openai_api(user_message_text)

            return final_response
    except Exception as e:
        return (f"error occured", e)


# --- Main Application Loop ---


def main():
    """Runs the command-line chat application."""
    user_id = str(uuid.uuid4())
    print("--- Conversational Memory Agent ---")
    print(f"Welcome! Your session ID is: {user_id}")
    print("Type 'quit' to exit.")
    createQdrant(user_id)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        print("Bot: Thinking...")

        response = handle_user_message(user_id, user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
