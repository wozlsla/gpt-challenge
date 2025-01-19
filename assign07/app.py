import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

import streamlit as st
from pathlib import Path
import json


st.set_page_config(
    page_title="QuizGPT",
    page_icon="./files/ham.ico",
)

st.title("Quiz GPT")


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# For question, 다양하게 구성
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
                
            Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
            
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
            Use (o) to signal the correct answer.
                
            Question examples:
                
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
                
            If context language is Korean, you should make it by Korean.
            The difficulty level of the problem is '{level}'.

            Your turn!
                            
            Context: {context}
        """,
        )
    ]
)


@st.cache_data(show_spinner="Making quiz...")
def quiz_chain(_docs, topic, level):
    chain = question_prompt | llm
    return chain.invoke({"context": _docs, "level": level})


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    topic = None
    openai_api_key = st.text_input("Input your OpenAI API Key")

    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf file.",
            type=["pdf", "docx", "txt"],
        )
        if file:
            docs = split_file(file)

    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            # retriever = WikipediaRetriever(top_k_results=5)
            retriever = WikipediaRetriever(top_k_results=5, lang="ko")
            with st.status("Searching..."):
                docs = retriever.get_relevant_documents(topic)

    st.markdown("---")
    level = st.selectbox("Quiz Level", ("EASY", "HRAD"))
    st.markdown("---")
    st.write("Github: https://github.com/wozlsla/gpt-challenge/tree/main/assign07")

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini-2024-07-18",
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[
                StreamingStdOutCallbackHandler(),
            ],
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )

        response = quiz_chain(docs, topic if topic else file.name, level)
        response = response.additional_kwargs["function_call"]["arguments"]

    with st.form("questions_form"):
        questions = json.loads(response)["questions"]
        question_count = len(questions)
        success_count = 0
        for idx, question in enumerate(questions):
            st.markdown(f'#### {idx+1}. {question["question"]}')
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                success_count += 1
            elif value is not None:
                st.error("Wrong!")
        if question_count == success_count:
            st.balloons()

        button = st.form_submit_button()
