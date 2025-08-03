## Note
This project is still under developement, kindly raise issues if you run into anything.

# LongTermMemory
This project is to build a long term memory functionalities to the openai or gemini api's using RAG and Semantic Search using Qdrant as vector DB and all-MiniLM-L6-v2 model for vector creation

You can implement different models based on your requirement and hardware!


# System Requirement
You need to install [Docker](https://www.docker.com/) (Docker Desktop for windows)<br>
4 Gb RAM to run Qdrant locally<br>
Good Internet Connectivity<br>
python>3.9<br>
[Qdrant](https://qdrant.tech)
# Setup
1. Once you have installed Docker go ahead and clone this repository
```bash
git clone git@github.com:sravsimh/LongTermMemory.git
```
2. now run in a venv:
```bash
pip install -r requirements.txt 
```
3. once all the dependencies are installed go to main.py and paste your api keys either gemini or openai 
```bash
OPENAI_API_KEY = "Your-API-KEY"

or 

os.environ["GEMINI_API_KEY"] = "your-api-key"
```

4. make sure to replace the
```python
call_gemini_api
```
with

```python
call_openai_api
```
or vice versa based on what API you are using

now open docker desktop and run:
```bash
docker pull qdrant/qdrant
```
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

this will start a local qdrant DB

now run:

```python
python main.py
```

this will run a chat session with a uuid and have the data stored for the same uuid

# Scaling
If you want to scale this and build a UI please use Qdrant Cloud and instead of creating collections with uuid maintain single collection and add UUID filter in payload

# Results
The below is the interaction i had with the agent

```bash
$ python main.py
--- Conversational Memory Agent ---
Welcome! Your session ID is: e68f851c-ba8b-413f-aa49-2892453d079f
Type 'quit' to exit.

You: i use notion and conda for productivity
Bot: Thinking...
added vector to qdrant DB
added ['uses Notion for productivity', 'uses Conda for productivity'] to memory
Bot: It looks like you’re already utilizing Notion and Conda for boosting your productivity! If you have any specific tips, tricks, or integrations you’d like to explore with these tools, let me know! For instance, I could help with setting up workflows in Notion or using Conda environments more efficiently.

You: can you tell me a short joke about ML devs
Bot: Thinking...
Bot: Why did the ML developer break up with their partner?

They needed more "data" to make a "well-informed" decision!

You: lolll, by the way can you forget that i use conda since i stopped using it
Bot: Thinking...
Deleted payload successfully
I have forgotten about uses Conda for productivity
Bot: Got it! I will forget that you used Conda. If there's anything else you'd like to update or change, just let me know!

You: quit
Goodbye!
(aimemoryagent)
```



<p align="center">made with ❤️ by --<strong>sravsimh</strong>--</p>