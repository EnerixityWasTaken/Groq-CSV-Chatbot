import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq

models = ["distil-whisper-large-v3-en","gemma2-9b-it","gemma-7b-it","llama-3.3-70b-versatile","llama-3.1-8b-instant","llama-guard-3-8b","llama3-70b-8192","llama3-8b-8192","mixtral-8x7b-32768","whisper-large-v3","whisper-large-v3-turbo"]


def return_csv_agent(user_query, csv_file, api_key,model):
    try:
        csv_file.seek(0)
        llm = ChatGroq(model=model, api_key=api_key)
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
    model = st.selectbox('Choose a model',models)
    if st.button("Choose Model"):
            if not model:
                st.warning("Please choose a model.")
                return
            # csv_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
            csv_file = st.file_uploader('Upload CSV file', type=['csv'])
            if csv_file:
                    # st.sidebar.subheader('CSV File')
                    # df = pd.read_csv(csv_file)
                    # st.sidebar.dataframe(df)
        
                    user_query = st.chat_input('Ask a question about your CSV file:')
                    if user_query:
                        with st.chat_message('user'):
                            st.write(user_query)
                        
                        try:
                            response = return_csv_agent(user_query=user_query, csv_file=csv_file, api_key=api_key,model = model)
                            with st.chat_message('assistant'):
                                st.write(response)
                        except ValueError as e:
                            with st.chat_message('assistant'):
                                st.error(f"Error: {e}")
            else:
                st.info("Upload a CSV file to get started.")
            

main()
