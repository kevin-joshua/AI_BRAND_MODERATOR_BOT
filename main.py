from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
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
from pathlib import Path
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from uuid import uuid4



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


BRAND_NAME = ""
UPLOAD_DIR = "uploaded_pdfs"
FILENAME = "brand_data.txt"
START_URL = "https://www.example.com"  # Replace with the actual URL
MAX_DEPTH = 4
MAX_PAGES = 50
MAX_TOKENS = 80000
visited = set()
session = requests.Session()
session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0 Safari/537.36"
    })

os.makedirs(UPLOAD_DIR, exist_ok=True)

# FUNCTION LOGIC


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

def scrape_page(url, depth):
  global visited


  if depth == 0:
      with open("brand_links.txt", "w", encoding='utf-8') as f:
          f.write(f"{url}\n")

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


def load_chunks():
    with open("chunked_data.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def initialize_vector_db(brand_name):
    """
    Initialize the vector database.
    """
    persist_directory = f"db/{brand_name}/"
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return db

def add_to_vector_db(db, lines, pdf_text):
    """
    Add new documents to the vector database.
    """
    # Load existing documents
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


def load_chroma(persist_directory):
    """
    Load the vector database.
    """
        
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    return db

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

def generate_response(content, user_query, brand_name):
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




app = FastAPI(
    title="Brand Moderator API",
    description="API for scraping websites and querying content using AI",
    version="1.0.0"
)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or "*" to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return {"message": "Welcome to the Brand Moderator API!"}


@app.post("/scrape")
async def get_brand_url(url: str):
    """
    Scrape a website and store the content in a vector database.
    """
    # Extract the brand name from the URL
    extracted = tldextract.extract(url)
    brand_name = extracted.domain
    global BRAND_NAME
    BRAND_NAME = brand_name
    START_URL = url
    depth = 0
    print(f"[+] Brand name extracted: {brand_name}")
    # Check if the URL is valid
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        raise HTTPException(status_code=400, detail="Invalid URL")

    # Start the scraping task in the background
    scrape_page(url, depth)
    print(f"[✓] Scraping started for {url}.")
    print(f"[✓] Scraping completed. Total pages scraped: {len(visited)}")
    print(f"[✓] Data saved to {FILENAME}.")

    with open("brand_links.txt", "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print(f"[✓] {line_count} links saved to brand_links.txt.")

    with open("visited.txt", "w", encoding="utf-8") as f:
        for link in visited:
            f.write(f"{link}\n")
    print(f"[✓] {len(visited)} links saved to visited.txt.")

    await process_data()

    return {"message": "Scraping Completed", "brand_name": brand_name, "url": url}

@app.post("/upload_pdf")
async def upload_pdf(pdfFiles: List[UploadFile] = File(...)):
    """
    Upload a PDF file and extract text from it.
    """
    if not pdfFiles.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    for file in pdfFiles:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed. '{file.filename}' is not a PDF.")

        try:
            # Save to uploaded_pdfs/ with a unique name to avoid conflicts
            filename = f"{uuid4().hex}_{file.filename}"
            save_path = os.path.join(UPLOAD_DIR, filename)

            with open(save_path, "wb") as out_file:
                content = await file.read()
                out_file.write(content)

            # Extract text from the saved PDF
            pdf_text = extract_text_from_pdf(save_path)

            

            if not pdf_text:
                raise HTTPException(status_code=400, detail="No text found in the PDF.")

            return {
                "filename": pdfFiles.filename,
                "text_preview": pdf_text[:500],  # optional preview
                "message": "✅ Text extracted successfully."
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    


async def process_data():

  # Initialize the vector database
  db = initialize_vector_db(BRAND_NAME)
  print(f"[✓] Vector DB initialized for {BRAND_NAME}.")


 
  # LOAD THE SCRAPED DATA
  data = load_data()

  # CHUNK THE DATA

  chunks = chunk_text_by_tokens(data, tokenizer)
  print(f"[✓] {len(chunks)} chunks created from the data.")  


  lines = load_chunks()
  print(f"[✓] {len(lines)} lines loaded from chunked_data.txt.") 

  # Load and process all saved PDFs
  pdf_text = []
  for filename in os.listdir("uploaded_pdfs"):
    if filename.endswith(".pdf"):
        file_path = os.path.join("uploaded_pdfs", filename)
        text = extract_text_from_pdf(file_path)
        pdf_text.extend(text)
        print(f"Processed {filename}: {len(text)} characters extracted.")

  add_to_vector_db(db, lines, pdf_text)

  print(f"[✓] Data added to the vector database.")



@app.get("/query")
async def query_data(brand: str, query: str):
    """
    Query the vector database and generate a response.
    """
    try:
        # Load the vector database
        chromaDB = load_chroma(f"db/{brand}/")
        print(f"[✓] Vector DB loaded for {brand}.")

        # Perform the query
        results = query_mmr(chromaDB, query)
        print(f"[✓] MMR search completed. Found {len(results)} results.")

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant content found. Please scrape the website first."
            )

        # Generate response
        response = generate_response(results, query, brand)

        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)

        if not response:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )

        

        return {
            "query": query,
            "answer": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))