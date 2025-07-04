import streamlit as st
import requests

st.set_page_config(page_title="Car Information Agent", page_icon="🚗",layout='centered')

st.title("Car Information Agent 🚗")

st.markdown("Ask anything about cars and get instant answers!")


question = st.text_input("Enter your question about cars:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"question": question}
                )
                if response.status_code == 200:
                    st.success("Answer:")
                    st.write(response.json()["response"])
                else:
                    st.error("Failed to get a response from the server. Please try again later.")
            except Exception as e:
                st.error(f"Error: {str(e)}")