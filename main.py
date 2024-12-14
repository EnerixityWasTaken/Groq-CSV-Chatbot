import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq

def return_csv_agent(query,csv_file):
    csv_file.seek(0)
    llm = ChatGroq(model = "llama-3.1-8b-instant",api_key = "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")
    agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)
    response = agent.run(query)
    return response
def main():
    st.set_page_config(page_title="CSV agent",page_icon=":robot_face:")
    st.title("CSV Agent")

    csv_file = st.file_uploader("Upload CSV file",type = ['csv'])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.sidebar.dataframe(df)
        query = st.chat_input("Ask a question about your CSV file")
        if query:
            response = return_csv_agent(query,csv_file)

            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                st.write(response)
    
main()

    
    
