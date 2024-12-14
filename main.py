import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq

def return_csv_agent(user_query, csv_file):
    csv_file.seek(0)
    llm = ChatGroq(model='llama-3.1-8b-instant')
    
    agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)
    
    response = agent.run(user_query)
    return response


def main():
    load_dotenv()
    
    st.set_page_config(page_title='CSV Agent', page_icon=':robot_face:', layout='wide')
    st.title('CSV Agent')
    
    
    csv_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    if csv_file:
        st.sidebar.subheader('CSV File')
        df = pd.read_csv(csv_file)
        st.sidebar.dataframe(df)
        user_query = st.chat_input('Ask a question about your CSV file')       
        if user_query:            
            with st.chat_message('user'):
                st.write(user_query)
            response = return_csv_agent(user_query=user_query, csv_file=csv_file)
            with st.chat_message('assistant'):
                st.write(response)



main()
