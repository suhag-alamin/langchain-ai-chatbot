import streamlit as st
from chatbot import ask
from langchain.messages import HumanMessage, AIMessage, SystemMessage


st.set_page_config(page_title="Langchain AI Chatbot", page_icon="🤖")

st.header("Langchain AI Chatbot")
st.caption("RunnableBranch + RunnableParallel + Pydantic structured output")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a AI assistant. "
        )
    ]


# show old messages
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue

    role = "assistant" if isinstance(msg, AIMessage) else "user"

    with st.chat_message(role):
        st.write(msg.content)

# print chat history
with st.sidebar:
    st.subheader("Chat History")
    if not st.session_state.messages:
        st.write("Empty")
    else:
        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage):
                continue
            role = "AI" if isinstance(msg, AIMessage) else "Human"
            st.write(f"**{role}:** {msg.content}")

    # clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [SystemMessage(
            content="You are a AI assistant. ")]
        st.rerun()

# user input box

user_input = st.chat_input("Ask me anything")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            category, result = ask(user_input)

        st.write(result.answer)

        # show the structured output
        st.write("**Category:**", category)
        st.write("**Summary:**", result.summary)
        st.write("**Keywords:**", ", ".join(result.keywords))
        st.write("**Difficulty:**", result.difficulty)
        st.write("**Confidence:**", result.confidence)

    st.session_state.messages.append(AIMessage(content=result.answer))
