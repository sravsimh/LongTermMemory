import os
import json
import uuid
import requests
from transformer import createEmbeddings
from vector_embeddings import createQdrant, addToQdrant, deleteQdrant, searchQdrant


# --- Configuration ---
# IMPORTANT: Set up your environment variables before running.
# 1. Get your Gemini API key from Google AI Studio.
# 2. Set the GEMINI_API_KEY environment variable.

os.environ["GEMINI_API_KEY"] = "AIzaSyDvvgkeL2cYRBDrW5ETtaREUtxqfwQvlnY"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"


# --- Helper function to call Gemini API ---


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
        Analyze the following user message. Does it express a desire to forget, remove, or update a piece of information?
        Message: "{user_message_text}"
        If yes, identify the core subject of the information to be forgotten.
        Respond in JSON format with two keys: "shouldDelete" (boolean) and "memoryToForget" (string, the subject to forget, or null).
        example: if user says im no longer using brave for browsing return {{"shouldDelete":"True","memoryToForget":["brave for browsing"]}}
    """

    deletion_response_text = call_gemini_api(deletion_prompt, is_json=True)
    try:
        deletion_response = json.loads(deletion_response_text)
        if deletion_response.get("shouldDelete") and isinstance(deletion_response.get("memoryToForget"), list):
            subject_to_forget = deletion_response["memoryToForget"]
            vectors = createEmbeddings(subject_to_forget)
            for i, vec in enumerate(vectors):
                searchResults = searchQdrant(user_id, vec)
                if (len(searchResults) > 0):
                    for res in searchResults:
                        deleteQdrant(user_id, res)
                        print(f"I have forgotten about {subject_to_forget[i]}")
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
    creation_response_text = call_gemini_api(creation_prompt, is_json=True)
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
    message: {user_message_text} and return {{"requiresMemory":"True"}} else {{"requiresMemory":"False"}} 
    '''
    casual_prompt_text_response = call_gemini_api(
        casual_prompt_text, is_json=True)

    try:
        casual_response = json.loads(casual_prompt_text_response)
        if casual_response.get("shouldRemember"):

            # 3. Generate a response using context and memories
            res = searchQdrant(user_id, createEmbeddings(user_message_text)[0])
            indexedMatches = []
            for i in res:
                indexedMatches.append(i.payload["content"])

            if len(indexedMatches) != 0:
                memories_context = "Here are some things I know about the user:\n- " + \
                    "\n- ".join(indexedMatches)
            else:
                memories_context = "I don't have any memories about the user yet."

            response_prompt = f"""
                You are a helpful assistant with a long-term memory.
                {indexedMatches}
                Based on this context and our current conversation, provide a helpful and relevant response to the user's latest message.
                User's message: "{user_message_text}"
            """
            final_response = call_gemini_api(response_prompt)

            return final_response
        else:
            final_response = call_gemini_api(user_message_text)

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
