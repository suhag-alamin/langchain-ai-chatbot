# LangChain AI Chatbot

A small Streamlit chatbot that routes each question to a different assistant and returns a structured result instead of plain text.

The point of this project is to show three LangChain ideas working together in one app:

- `RunnableBranch` to pick the right prompt for the question
- `RunnableParallel` to run two chains at the same time
- Pydantic models to force the model output into a fixed shape

## Project Overview

When you ask a question, the app does four things:

1. Classifies the question as `programming`, `math`, or `general`
2. Sends it to the matching prompt
3. Writes a one line summary at the same time
4. Merges the answer and the summary into a single Pydantic object

The UI then shows the answer along with the category, summary, keywords, difficulty, and a confidence score.

Here is the flow:

```
question
   |
   v
classifier_chain  ->  category
   |
   v
parallel_chain
   |-- branch_chain  --> answer   (programming / math / general prompt)
   |-- summary_chain --> summary
   |
   v
final_chain  ->  ChatResponse
```

## Features

- Question routing across three assistants
- Two chains running in parallel instead of one after the other
- Typed output validated by Pydantic
- Chat history kept in the sidebar
- Clear chat button
- Groq as the model provider, so responses come back fast

## RunnableBranch Implementation

`RunnableBranch` works like an if / elif / else for chains. Each condition is a function that reads the input dict, and the last chain is the fallback when nothing matches.

From `chatbot.py`:

```python
branch_chain = RunnableBranch(
    (lambda x: x["category"] == "programming", programming_chain),
    (lambda x: x["category"] == "math", math_chain),
    general_chain
)
```

The `category` value comes from the classifier step that runs before this. Each of the three chains uses its own prompt from `prompts.py` but shares the same `answer_model`, so a math question gets a step by step tutor and a coding question gets an explanation with a code sample.

There is also a guard in `ask()` so a bad classification cannot break the branch:

```python
if category not in ["programming", "math", "general"]:
    category = "general"
```

## RunnableParallel Implementation

`RunnableParallel` takes one input and feeds it to several chains at once. The answer and the summary do not depend on each other, so there is no reason to wait for the first before starting the second.

```python
parallel_chain = RunnableParallel({
    "answer": branch_chain,
    "summary": summary_chain
})
```

Both chains receive the same dict:

```python
result = parallel_chain.invoke({
    "question": question,
    "category": category
})
```

The output is a dict with the same keys, so `result["answer"]` is an `Answer` object and `result["summary"]` is a `Summary` object.

## Pydantic Structured Output Implementation

Every model call in this project returns a validated object, never a raw string. The schemas live in `schemas.py`:

```python
class ChatResponse(BaseModel):
    answer: str = Field("The final answer for the user")
    summary: str = Field("A one line summary of the answer")
    keywords: List[str] = Field("3 to 5 important keywords")
    difficulty: str = Field("Easy, Medium or Hard")
    confidence: float = Field(
        ge=0,
        le=1,
        description="How confident the answer is, between 0 and 1"
    )
```

Each schema is bound to the model with `with_structured_output`:

```python
route_model = model.with_structured_output(Route, method="json_schema")
answer_model = model.with_structured_output(Answer, method="json_schema")
summary_model = model.with_structured_output(Summary, method="json_schema")
final_model = model.with_structured_output(ChatResponse, method="json_schema")
```

`method="json_schema"` sends the schema to the API so the model is constrained while generating. That is stricter than asking for JSON in the prompt and hoping for the best. It also means `confidence` is guaranteed to be a float between 0 and 1, so the UI can read `result.confidence` without any parsing or try / except.

Note that `json_schema` is not supported by every Groq model. The `openai/gpt-oss-*` models work. The Llama and Qwen models on Groq reject it, so switching the model name alone will break the app.

## Installation

You need Python 3.10 or newer and a free Groq API key from https://console.groq.com.

Clone the project and move into it:

```bash
git clone https://github.com/suhag-alamin/langchain-ai-chatbot.git
cd langchain-ai-chatbot
```

Install the packages:

```bash
pip install -r requirements.txt
```

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Then open `.env` and fill it in:

```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Project Structure

```
langchain-ai-chatbot/
├── app.py            Streamlit UI and session state
├── chatbot.py        chains, model setup, ask()
├── prompts.py        prompt templates for every step
├── schemas.py        Pydantic output models
├── requirements.txt
└── .env.example
```

## Configuration

The model is set in one place, at the top of `chatbot.py`:

```python
model = ChatGroq(model="openai/gpt-oss-120b")
```

`openai/gpt-oss-20b` is the lighter option if you start hitting the free tier token limit. The Groq free tier allows 1000 requests per day but only 8000 tokens per minute, and this app uses four calls per question, so a burst of questions can hit that limit before the daily one.

The system prompt shown in the chat comes from `app.py`:

```python
SystemMessage(content="You are a AI assistant. ")
```

## Known Limitations

- The chat history is displayed but not sent back to the model. Each question is answered on its own, so follow up questions like "explain that again" will not work yet.
- History lives in Streamlit session state, so it is gone when you refresh the page.
- Running the app may print a `torch._classes` RuntimeError in the terminal. That comes from the Streamlit file watcher, not from this code. Use `streamlit run app.py --server.fileWatcherType none` to silence it.
