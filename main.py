import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from transformers import AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import hashlib
from groq import Groq
import tldextract
import streamlit as st
from pathlib import Path
import tempfile


# Streamlit app

st.set_page_config(page_title="Brand Bot", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"



load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize client

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)



# ---- Config ----
START_URL = "https://thehouseofrare.com/"
MAX_DEPTH = 4
MAX_PAGES = 50
FILENAME = "brand_data.txt"
MAX_TOKENS = 80000


extracted = tldextract.extract(START_URL)
brand_name = extracted.domain



visited = set()
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0 Safari/537.36"
})

# ---- Utils ----
def is_internal(url, base_url):
    base = urlparse(base_url).netloc
    target = urlparse(url).netloc
    return target == base or target.endswith("." + base)

def is_valid_url(url):
    return not url.lower().endswith((
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.pdf', '.zip'
    ))

def is_priority_url(url):
    # Optional: restrict to product/about/faq/contact pages
    return any(key in url.lower() for key in ['product', 'shop', 'about', 'faq', 'contact', 'support', 'help'])

def safe_get(url, retries=2, delay=2):
    for _ in range(retries):
        try:
            return session.get(url, timeout=10)
        except requests.RequestException as e:
            print(f"[!] Request failed: {e}")
            time.sleep(delay)
    return None

# ---- Core Scraper ----
def scrape_page(url, depth):
    global visited

    if depth == 0:
        with open('brand_links.txt', 'w', encoding="utf-8") as f:
            f.write(f" ")

    if depth > MAX_DEPTH or len(visited) >= MAX_PAGES or url in visited:
        return

    print(f"[+] Scraping: {url} (Depth: {depth})")
    visited.add(url)

    response = safe_get(url)
    if not response or response.status_code != 200:
        print(f"[!] Failed to load {url}")
        return

    soup = BeautifulSoup(response.text, 'lxml')
    if not soup:
        print(f"[!] Failed to parse {url}")
        return

    # Remove junk tags
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    content = soup.get_text(separator=' ', strip=True)

    if depth == 0:
        mode = "w"
    else:
        mode = "a"

    with open(FILENAME, mode, encoding="utf-8") as f:
        f.write(content + "\n\n")

    

   
    for a_tag in soup.find_all('a', href=True):
        href = urljoin(url, a_tag['href']).split('#')
        for link in href:
            with open("brand_links.txt", 'a', encoding="utf-8") as f:
                f.write(f"[+] : {link}\n")
            if is_priority_url(link):
                scrape_page(link, depth + 1)
            elif is_internal(link, START_URL) and is_valid_url(link):
                scrape_page(link, depth + 1)
        

# ---- Start ----

def extract_text_from_pdf(pdf_path):
    try:
        
            loader = PyPDFLoader(pdf_path)
            text = loader.load_and_split()
            print(f"[✓] PDF loaded successfully.")
            if not text:
                print("[!] No text found in the PDF.")
                return []
            return text
    except Exception as e:
        print(f"[!] Error loading PDF: {e}")
        return []



def load_data():
    with open("brand_data.txt", "r", encoding="utf-8") as f:
        return f.read()


def chunk_text_by_tokens(text, tokenizer):
    '''
    Splits the text into chunks based on token count.
    '''
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function= lambda x: len(tokenizer(x)["input_ids"]),
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"[✓] Text chunked into {len(chunks)} parts.")

    with open("chunked_data.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            cleaned_chunk = " ".join(chunk.strip().split())
            f.write(f"{cleaned_chunk}\n")
    print(f"[✓] Chunked data saved to chunked_data.json.")
    return chunks

data = load_data()

chunks = chunk_text_by_tokens(data, tokenizer)
print(f"[✓] {len(chunks)} chunks created from the data.")


def load_chunks():
    with open("chunked_data.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
lines = load_chunks()
print(f"[✓] {len(lines)} lines loaded from chunked_data.txt.")

pdf_path = "thehouseofrare.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

persist_directory = f"db/{brand_name}/"
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


existing_hashes = set()
existing_docs = db.get()
for meta in existing_docs["metadatas"]:
    if meta and "hash" in meta:
        existing_hashes.add(meta["hash"])

# Prepare new documents
new_docs = []
for line in lines:
    chunk = line.strip()
    chunk_hash = get_hash(chunk)

    if chunk_hash not in existing_hashes:
        new_docs.append(Document(page_content=chunk, metadata={"hash": chunk_hash}))
        existing_hashes.add(chunk_hash) 

for line in pdf_text:
    chunk = line.page_content.strip()
    chunk_hash = get_hash(chunk)

    if chunk_hash not in existing_hashes:
        new_docs.append(Document(page_content=chunk, metadata={"hash": chunk_hash}))
        existing_hashes.add(chunk_hash)
        print(f"[✓] New chunk added from pdf: {chunk_hash}")
# Add only new documents to Chroma
if new_docs:
    db.add_documents(new_docs)
    print(f"[✓] Added {len(new_docs)} unique chunks to Chroma.")
else:
    print("[✓] No new unique chunks to add.") 


def load_chroma():
    
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    return db


chroma_db = load_chroma()
print("[✓] Chroma DB loaded.")

def estimate_tokens(text):
    return int(len(text.split()) * 1.3)  


def query_mmr(db, query, k=4, fetch_k=15, lambda_mult=0.5):
    """
    Perform MMR search on Chroma vector DB.
    """
    formatted_query = f"query: {query}"
    try:
        results = db.max_marginal_relevance_search(
            formatted_query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        print(f"[✓] {len(results)} MMR results found.")
        return results
    except Exception as e:
        print(f"[!] MMR search failed: {e}")
        return []

def generate_response(content, user_query):
    '''
    Generates a response using the Groq API.
    '''


    if not content:
        print("[!] No content to analyze.")
        return None
    
    combined = ""
    total_tokens = 0

    for doc in content:
        chunk = doc.page_content.strip()
        chunk_tokens = estimate_tokens(chunk)
        if total_tokens + chunk_tokens > MAX_TOKENS:
            break
        combined += f"\n\n{chunk}"
        total_tokens += chunk_tokens
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # exact model id on Groq
            messages=[
            {"role": "system", "content": f'''
                                            You are a knowledgeable and helpful AI assistant specializing in analyzing and interpreting content related to the brand "{brand_name}". 

                                            - Provide clear, well-structured, and informative answers to the user's query in a natural, confident tone.
                                            - Speak as if you already know the information—never mention the source, website, or that content was provided.
                                            - If multiple relevant facts or points are found, include all of them in the response.
                                            - Only answer if the question is relevant to that brand and supported by the information.
                                            - Do not make assumptions or speculate about additional products, services, or facts not explicitly mentioned.
                                            - Do not reference or imply the existence of information outside the available material (e.g., "Brand may offer more products").
                                            - If the user's question is unrelated to the brand or lacks supporting information, respond naturally and confidently that no relevant information is available—without referring to the data or content itself.
                                            - Avoid filler, guesses, or generalizations. Stick to what's clearly stated, but explain it in full where possible.

                                            '''


             },
            {"role": "user", "content": f" user_query:{user_query}\n\n content:{combined.strip()}"},
            ],
            temperature=0.3
        )
        return response
    except Exception as e:
        print(f"[!] Error generating response: {e}")
        return None




# query = "Tell me what is the cheapest product available on the website?"
# query = "Tell me about the brand and its products? Also, where can I find the products?"
# query = "Tell me about the offers that the brand is currently running? Tell me all about the offers and which banks are providing the offers?"
# results = query_mmr(db, query)
# response = generate_response(results, query)
# if response:
#     print(f"[✓] Response: {response.choices[0].message.content}")
#     with open('response_tracking.txt', 'a', encoding="utf-8") as f:
#         f.write(f"----------------------------START-------------------------------\n")
#         f.write(f"URL: {START_URL}\n")
#         f.write(f" -> MAX PAGES: {MAX_PAGES}\n")
#         f.write(f" -> MAX DEPTH: {MAX_DEPTH}\n")
#         f.write(f"[?] Query: {query}\n")
#         f.write(f"[A] Response: {response.choices[0].message.content}\n")
#         f.write("----------------------------END--------------------------------\n")
#         print(f"[✓] Response saved to response_tracking.txt.\n")
# else:
#     print("[!] Failed to generate response.")







if st.session_state.page == "home":
    st.title("🛍️ Brand Data Ingestor")
    st.write("Enter a website URL and/or upload PDF files.")

    url = st.text_input("Enter Website URL:")
    pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Submit"):
        st.write("Processing...")

        START_URL = url
        scrape_page(START_URL, depth=0)
        print(f"[✓] Scraping completed. Total pages scraped: {len(visited)}")
        print(f"[✓] Data saved to {FILENAME}.")
        with open("brand_links.txt", "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"[✓] {line_count} links saved to brand_links.txt.")

        with open("visited.txt", "w", encoding="utf-8") as f:
            for link in visited:
                f.write(f"{link}\n")
        print(f"[✓] {len(visited)} links saved to visited.txt.")




        pdf_paths = []
        for uploaded_file in pdfs:
            if uploaded_file is not None:
                # Write uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                    pdf_paths.append(tmp_file_path)
        for pdf_path in pdf_paths:
            pdf_text = extract_text_from_pdf(pdf_path)
        st.success("✅ Data added successfully!")

        st.session_state.page = "bot"

elif st.session_state.page == "bot":
    st.title("💬 Chat with Brand Bot")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
    query = st.text_input("Ask something about the brand:")

    if query:
        with st.spinner("Thinking..."):
            results = query_mmr(db, query)
            response = generate_response(results, query)
            if response:
                print(f"[✓] Response: {response.choices[0].message.content}")
                with open('response_tracking.txt', 'a', encoding="utf-8") as f:
                    f.write(f"----------------------------START-------------------------------\n")
                    f.write(f"URL: {START_URL}\n")
                    f.write(f" -> MAX PAGES: {MAX_PAGES}\n")
                    f.write(f" -> MAX DEPTH: {MAX_DEPTH}\n")
                    f.write(f"[?] Query: {query}\n")
                    f.write(f"[A] Response: {response.choices[0].message.content}\n")
                    f.write("----------------------------END--------------------------------\n")
                    print(f"[✓] Response saved to response_tracking.txt.\n")
            else:
                print("[!] Failed to generate response.")
            
            st.write("🤖", f"{response.choices[0].message.content}")  
    