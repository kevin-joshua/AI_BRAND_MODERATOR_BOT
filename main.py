import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from transformers import AutoTokenizer
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

# Global variable for brand name
brand_name = None

# os.environ["STREAMLIT_FOLDER_WATCH_BLACKLIST"] = "venv,torch"

# ---- Utils ----
def is_internal(url, base_url):
    with st.spinner("Checking if URL is internal..."):
        base = urlparse(base_url).netloc
        target = urlparse(url).netloc
        return target == base or target.endswith("." + base)

def is_valid_url(url):
    with st.spinner("Checking if URL is valid..."):
        return not url.lower().endswith((
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.pdf', '.zip'
        ))

def is_priority_url(url):
    with st.spinner("Checking if URL is a priority..."):
        return any(key in url.lower() for key in ['product', 'shop', 'about', 'faq', 'contact', 'support', 'help'])

def safe_get(url, session, retries=2, delay=2):
    with st.spinner(f"Requesting URL: {url}"):
        for _ in range(retries):
            try:
                return session.get(url, timeout=10)
            except requests.RequestException as e:
                print(f"[!] Request failed: {e}")
                time.sleep(delay)
        return None

def scrape_page(url, depth, visited, session, START_URL, MAX_DEPTH, MAX_PAGES, FILENAME):
    with st.spinner(f"Scraping page: {url} at depth {depth}"):
        if depth == 0:
            with open('brand_links.txt', 'w', encoding="utf-8") as f:
                f.write(f" ")
        if depth > MAX_DEPTH or len(visited) >= MAX_PAGES or url in visited:
            return
        print(f"[+] Scraping: {url} (Depth: {depth})")
        visited.add(url)
        response = safe_get(url, session)
        if not response or response.status_code != 200:
            print(f"[!] Failed to load {url}")
            return
        soup = BeautifulSoup(response.text, 'lxml')
        if not soup:
            print(f"[!] Failed to parse {url}")
            return
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        content = soup.get_text(separator=' ', strip=True)
        mode = "w" if depth == 0 else "a"
        with open(FILENAME, mode, encoding="utf-8") as f:
            f.write(content + "\n\n")
        for a_tag in soup.find_all('a', href=True):
            href = urljoin(url, a_tag['href']).split('#')
            for link in href:
                with open("brand_links.txt", 'a', encoding="utf-8") as f:
                    f.write(f"[+] : {link}\n")
                if is_priority_url(link):
                    scrape_page(link, depth + 1, visited, session, START_URL, MAX_DEPTH, MAX_PAGES, FILENAME)
                elif is_internal(link, START_URL) and is_valid_url(link):
                    scrape_page(link, depth + 1, visited, session, START_URL, MAX_DEPTH, MAX_PAGES, FILENAME)

def extract_text_from_pdf(pdf_path):
    with st.spinner(f"Extracting text from PDF: {pdf_path}"):
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

def load_data(filename):
    with st.spinner(f"Loading data from file: {filename}"):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text_by_tokens(text, tokenizer):
    with st.spinner("Chunking text by tokens..."):
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
        print(f"[✓] Chunked data saved to chunked_data.txt.")
        return chunks

def load_chunks():
    with st.spinner("Loading chunked data from file..."):
        with open("chunked_data.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

def get_hash(text):
    with st.spinner("Hashing text chunk..."):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

def estimate_tokens(text):
    with st.spinner("Estimating token count..."):
        return int(len(text.split()) * 1.3)

def query_mmr(db, query, k=4, fetch_k=15, lambda_mult=0.5):
    with st.spinner("Querying Chroma DB with MMR..."):
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

def generate_response(client, brand_name, content, user_query, MAX_TOKENS):
    with st.spinner("Generating response from LLM..."):
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
                model="llama3-8b-8192",
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
                        - Include source references when appropriate.
                    '''},
                    {"role": "user", "content": f" user_query:{user_query}\n\n content:{combined.strip()}"},
                ],
                temperature=0.3
            )
            return response
        except Exception as e:
            print(f"[!] Error generating response: {e}")
            return None

def main():
    global brand_name
    st.set_page_config(page_title="Brand Bot", layout="wide")
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "visited" not in st.session_state:
        st.session_state.visited = set()
    if "db" not in st.session_state:
        st.session_state.db = None
    if "brand_name" not in st.session_state:
        st.session_state.brand_name = None
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "client" not in st.session_state:
        st.session_state.client = None
    if "persist_directory" not in st.session_state:
        st.session_state.persist_directory = None
    if "MAX_DEPTH" not in st.session_state:
        st.session_state.MAX_DEPTH = 4
    if "MAX_PAGES" not in st.session_state:
        st.session_state.MAX_PAGES = 50
    if "FILENAME" not in st.session_state:
        st.session_state.FILENAME = "brand_data.txt"
    if "MAX_TOKENS" not in st.session_state:
        st.session_state.MAX_TOKENS = 80000

    load_dotenv()

    if st.session_state.page == "home":
        st.title("🛍️ AI COMMUNITY MODERATOR")
        st.write("Enter a website URL and/or upload PDF files.")
        url = st.text_input("Enter Website URL:")
        pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit"):
            st.write("Processing...")
            START_URL = url
            st.session_state.START_URL = START_URL
            extracted = tldextract.extract(START_URL)
            brand_name = extracted.domain
            st.session_state.brand_name = brand_name
            st.session_state.visited = set()
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0 Safari/537.36"
            })
            scrape_page(START_URL, 0, st.session_state.visited, session, START_URL, st.session_state.MAX_DEPTH, st.session_state.MAX_PAGES, st.session_state.FILENAME)
            print(f"[✓] Scraping completed. Total pages scraped: {len(st.session_state.visited)}")
            print(f"[✓] Data saved to {st.session_state.FILENAME}.")
            with open("brand_links.txt", "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"[✓] {line_count} links saved to brand_links.txt.")
            with open("visited.txt", "w", encoding="utf-8") as f:
                for link in st.session_state.visited:
                    f.write(f"{link}\n")
            print(f"[✓] {len(st.session_state.visited)} links saved to visited.txt.")
            pdf_paths = []
            for uploaded_file in pdfs:
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                        pdf_paths.append(tmp_file_path)
            all_pdf_text = []
            for pdf_path in pdf_paths:
                pdf_text = extract_text_from_pdf(pdf_path)
                all_pdf_text.extend(pdf_text)
                st.write(f"PDF text from uploaded files added to all_pdf_text. Current length: {len(all_pdf_text)}")
            st.session_state.all_pdf_text = all_pdf_text
            st.success("✅ Data added successfully!")
            st.session_state.page = "bot"
            # Model and embedding setup
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            st.session_state.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                encode_kwargs={"normalize_embeddings": True}
            )
            st.session_state.client = Groq(api_key=os.environ.get("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"])
            # Chunk and embed
            data = load_data(st.session_state.FILENAME)
            chunks = chunk_text_by_tokens(data, st.session_state.tokenizer)
            with open("chunked_data.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            st.session_state.lines = lines
            persist_directory = f"db/{brand_name}/"
            st.session_state.persist_directory = persist_directory
            db = Chroma(persist_directory=persist_directory, embedding_function=st.session_state.embedding_model)
            existing_hashes = set()
            existing_docs = db.get()
            for meta in existing_docs["metadatas"]:
                if meta and "hash" in meta:
                    existing_hashes.add(meta["hash"])
            new_docs = []
            for line in lines:
                chunk = line.strip()
                chunk_hash = get_hash(chunk)
                if chunk_hash not in existing_hashes:
                    new_docs.append(Document(page_content=chunk, metadata={"hash": chunk_hash}))
                    existing_hashes.add(chunk_hash)
            for line in all_pdf_text:
                chunk = line.page_content.strip()
                chunk_hash = get_hash(chunk)
                if chunk_hash not in existing_hashes:
                    new_docs.append(Document(page_content=chunk, metadata={"hash": chunk_hash}))
                    existing_hashes.add(chunk_hash)
                    st.spinner(f"New chunk added from pdf: {chunk_hash}")
                    print(f"[✓] New chunk added from pdf: {chunk_hash}")
            if new_docs:
                db.add_documents(new_docs)
                print(f"[✓] Added {len(new_docs)} unique chunks to Chroma.")
            else:
                print("[✓] No new unique chunks to add.")
            st.session_state.db = db
            st.rerun()
    elif st.session_state.page == "bot":
        st.title(f"💬 Chat with the Brand Bot")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "home"
        query = st.text_input("Ask something about the brand:")
        if query:
            with st.spinner("Thinking..."):
                db = st.session_state.db
                client = st.session_state.client
                brand_name = st.session_state.brand_name
                MAX_TOKENS = st.session_state.MAX_TOKENS
                results = query_mmr(db, query)
                response = generate_response(client, brand_name, results, query, MAX_TOKENS)
                if response:
                    print(f"[✓] Response: {response.choices[0].message.content}")
                    with open('response_tracking.txt', 'a', encoding="utf-8") as f:
                        f.write(f"----------------------------START-------------------------------\n")
                        f.write(f"URL: {st.session_state.START_URL}\n")
                        f.write(f" -> MAX PAGES: {st.session_state.MAX_PAGES}\n")
                        f.write(f" -> MAX DEPTH: {st.session_state.MAX_DEPTH}\n")
                        f.write(f"[?] Query: {query}\n")
                        f.write(f"[A] Response: {response.choices[0].message.content}\n")
                        f.write("----------------------------END--------------------------------\n")
                        print(f"[✓] Response saved to response_tracking.txt.\n")
                else:
                    print("[!] Failed to generate response.")
                st.write("🤖", f"{response.choices[0].message.content}")

if __name__ == "__main__":
    main()  