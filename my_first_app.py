import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

def select_model():
    st.sidebar.title("Options")
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0,step=0.1)
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown("**Total cost**")
    for i in range(3):
        st.sidebar.markdown(f"- ${i+0.01}")
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo-0613"
    else:
        model_name = "gpt-4"

    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []

def init_page():
    st.sidebar.title("Options")
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0,step=0.1)
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown("**Total cost**")
    for i in range(3):
        st.sidebar.markdown(f"- ${i+0.01}")
def main():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")

    llm = select_model()
    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")
if __name__ == '__main__':
    main()
