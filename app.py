import streamlit as st

# Home Page
st.title("Community Moderator AI")
st.write("This is a simple Streamlit app to demonstrate the Community Moderator AI.")

# Input Options
st.write("### Input Options:")
uploaded_file = st.file_uploader("Upload brand document", type=["pdf", "docx", "txt"])
url_input = st.text_input("Enter the brand page URL:")

# Enter Button
if st.button("Enter"):
    # Navigate to the next page
    st.write("### Processing...")
    with st.spinner("Chunking the data..."):
        import time
        time.sleep(2)  # Simulate processing time

    # Query Input
    st.write("### Enter Your Query:")
    query = st.text_input("Type your query here:")
    if st.button("Submit Query"):
        st.write(f"Processing query: {query}")