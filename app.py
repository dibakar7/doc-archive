import os
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File, Query
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Document(BaseModel):
    filename: str

class SearchResponse(BaseModel):
    documents: List[Document]


load_dotenv()


app = FastAPI()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


embeddings_model = SentenceTransformerEmbeddings(model_name = "all-MiniLm-L6-v2")



# Function to generate embeddings
def generate_embeddings(chunked_texts):
    vector_embeddings = embeddings_model.embed_documents(chunked_texts)
    return vector_embeddings



# function for text extraction
def extractText_from_PDF(pdf_file: UploadFile):
    import pdfplumber
    with pdfplumber.open(pdf_file.file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text



# pinecone initialization
pc = Pinecone(api_key=pinecone_api_key)
index_name = "doc-archive"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)




#split text into chunks
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_texts = text_splitter.split_text(text)
    return chunked_texts




# uploading doc file at '/upload'
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    text = extractText_from_PDF(file)
    chunked_texts = chunk_text(text = text)
    vector_embeddings = generate_embeddings(chunked_texts=chunked_texts)
    metadata = {"filename": file.filename}


    to_upsert = [(f"{file.filename}_{i}", vector_embeddings[i], metadata) for i in range(len(vector_embeddings))]
    index.upsert(vectors=to_upsert)

    return {"message": "File uploaded successfully!"}




# searching using query text
@app.get("/docs/", response_model=SearchResponse)
async def search_docs(query: str = Query(...)):
    query_embedding = embeddings_model.embed_query(query)
    search_results = index.query(vector=query_embedding, include_metadata=True, top_k=50)
    
    seen_filenames = set()
    document = []
    for match in search_results.matches:
        filename = match.metadata['filename']
        if filename not in seen_filenames and len(document)<3:          #To return top 3 distict file name
            seen_filenames.add(filename)
            document.append({"filename": filename})
    
    return {"documents": document}


#
if __name__ == "__app__":
    uvicorn.run(app, host="0.0.0.0", port=8000)






