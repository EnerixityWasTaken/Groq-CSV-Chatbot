import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq


def return_csv_agent(user_query, csv_file, api_key):
    try:
        csv_file.seek(0)
        llm = ChatGroq(model='llama-3.1-8b-instant', api_key=api_key)
        agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)
        response = agent.run(user_query)
        return response
    except Exception as e:
        raise ValueError(f"Error in processing: {e}")


def main():
    st.set_page_config(page_title='CSV Agent', page_icon=':robot_face:', layout='wide')
    st.title('CSV Agent')
    st.write("Ask questions about your CSV files using natural language.")

    api_key = st.text_input('Enter your Groq API key:', type="password")
    if not api_key:
        st.warning("Please enter your API key to use the application.")
        return

    csv_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    if csv_file:
        try:
            st.sidebar.subheader('CSV File')
            df = pd.read_csv(csv_file)
            st.sidebar.dataframe(df)

            user_query = st.chat_input('Ask a question about your CSV file:')
            if user_query:
                with st.chat_message('user'):
                    st.write(user_query)
                
                try:
                    response = return_csv_agent(user_query=user_query, csv_file=csv_file, api_key=api_key)
                    with st.chat_message('assistant'):
                        st.write(response)
                except ValueError as e:
                    with st.chat_message('assistant'):
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Failed to process the CSV file: {e}")
    else:
        st.info("Upload a CSV file to get started.")

main()
