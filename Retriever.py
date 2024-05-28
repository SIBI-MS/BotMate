from typing import List
import streamlit as st
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever


try:
    from langchain_community.llms import Ollama
except ImportError:
        print("If Ollama is not installed, please run 'curl -fsSL https://ollama.com/install.sh | sh")


def main():
    def saver(document_vector_store):
        st.session_state.vector_store = document_vector_store

    # Output parser will split the LLM result into a list of queries
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return lines

    output_parser = LineListOutputParser()


    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )




    # Initialize the callback manager
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Set the temperature parameter
    parameters = {
        "temperature": 0  # Adjust the temperature value as needed
    }

    # Initialize the Ollama model with the specified parameters
    llm = Ollama(
        model="phi3",
        parameters=parameters,
        callback_manager=callback_manager
    )

    llm_chain = QUERY_PROMPT | llm | output_parser

    # Run
    retriever = MultiQueryRetriever(
        retriever=st.session_state.document_vector_store.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    # Results
    def invoker(user_query):
        unique_docs = retriever.invoke(user_query)
        return unique_docs
    

if __name__ == "__main__":
    main()
