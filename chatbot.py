from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableBranch, RunnableParallel

from prompts import (
    classifier_prompt,
    programming_prompt,
    math_prompt,
    general_prompt,
    summary_prompt,
    final_prompt
)

from schemas import Route, Answer, Summary, ChatResponse

load_dotenv()

# model
model = ChatGroq(model="openai/gpt-oss-120b")


route_model = model.with_structured_output(Route, method="json_schema")
answer_model = model.with_structured_output(Answer, method="json_schema")
summary_model = model.with_structured_output(Summary, method="json_schema")
final_model = model.with_structured_output(ChatResponse, method="json_schema")

# classify chain
classifier_chain = classifier_prompt | route_model

# chain for each category
programming_chain = programming_prompt | answer_model
math_chain = math_prompt | answer_model
general_chain = general_prompt | answer_model

# runnablebranch

branch_chain = RunnableBranch(
    (lambda x: x["category"] == "programming", programming_chain),
    (lambda x: x["category"] == "math", math_chain),
    general_chain
)

# runnable parallel

summary_chain = summary_prompt | summary_model

parallel_chain = RunnableParallel({
    "answer": branch_chain,
    "summary": summary_chain
})

# final chain

final_chain = final_prompt | final_model


def ask(question):
    route = classifier_chain.invoke({
        "question": question
    })
    category = route.category.strip().lower()

    if category not in ["programming", "math", "general"]:
        category = "general"

    result = parallel_chain.invoke({
        "question": question,
        "category": category
    })

    final_result = final_chain.invoke({
        "answer": result["answer"].answer,
        "summary": result["summary"].summary
    })

    return category, final_result
