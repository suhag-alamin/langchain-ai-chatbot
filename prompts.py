from langchain_core.prompts import PromptTemplate

# step 1 --> decide which assistant should answer
classifier_prompt = PromptTemplate(
    template="""
    You are a question classifier.
    Classify the question below into one of these categories:
    - programming
    - math
    - general

    Put the chosen category in the "category" field.

    Question: {question}
    """,
    input_variables=["question"]
)

# step 2 --> one prompt for each category
programming_prompt = PromptTemplate(
    template="""
    You are a helpful Programming Assistant.
    Answer the question with a short explanation and a small code example. 
    Put your full answer in the "answer" field.

    Question: {question}
    """,
    input_variables=["question"]
)

math_prompt = PromptTemplate(
    template="""
    You are a friendly Math tutor.
    Solve the question step by step in simple words.
    Put your full answer in the "answer" field. 

    Question: {question}
    """,
    input_variables=["question"]
)

general_prompt = PromptTemplate(
    template="""
    You are a helpful general assistant. 
    Answer the question in simple and clear language. 
    Put your full answer in the "answer" field.

    Question: {question}
    """,
    input_variables=["question"]
)

# step 3 --> runs at the same time as the answer

summary_prompt = PromptTemplate(
    template="""
    Write a one line summary that explains what this question is about. 
    Put it in the "summary" field.

    Question: {question}
    """,
    input_variables=["question"]
)

# step 4 --> merge the parallel outputs into pydantic schema

final_prompt = PromptTemplate(
    template="""
    Below there is an answer and a summary.

    Answer: {answer}

    Summary: {summary}

    Fill the fields like this:
    - answer -> copy only the answer text above, nothing else
    - summary -> copy only the summary text above, nothing else
    - keywords -> 3 to 5 keywords about the topic
    - difficulty -> Easy, Medium or Hard
    - confidence -> a number between 0 and 1
    """,
    input_variables=["answer", "summary"]
)
