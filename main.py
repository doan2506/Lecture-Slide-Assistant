import pickle
import time
import os
from pathlib import Path
from tkinter.messagebox import QUESTION
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM = 'HuggingFaceH4/zephyr-7b-beta'
EXPORT_TYPE = ExportType.DOC_CHUNKS
TOP_K = 3
PROMPT_TEMPLATE = PromptTemplate.from_template(
    'Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query with under 300 words.\nQuery: {input}\nAnswer:\n'
)

data = Path('./data')
cache = Path('./cache')
cache.mkdir(exist_ok=True)

def metadata_simplify(metadata):
    dl_meta = metadata.get('dl_meta', None)
    headings = dl_meta.get('headings', None)
    doc_items = dl_meta.get('doc_items', None)
    provs = [item.get('prov', None) for item in doc_items]
    pages = [prov[0].get('page_no', None) for prov in provs]
    headings = dl_meta.get('headings', None)
    source = metadata.get('source', None)

    return {
        'page': list(set(pages)),
        'headings': headings,
        'source': source
    }

def load_documents():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print('Loading documents...')
    
    start_time = time.time()
    docling_imported = False 
    chunks = []

    tokenizer = HuggingFaceTokenizer(
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    )

    for subject in data.iterdir():
        for document in subject.iterdir():
            cached_doc = cache / f'{document.stem}.pkl'
            if cached_doc.exists():
                with open(cached_doc, 'rb') as f:
                    content = pickle.load(f)
                chunks.extend(content)
                continue
            if not docling_imported:
                docling_imported = True
                from langchain_docling import DoclingLoader
                from docling.chunking import HybridChunker
            try:
                loader = DoclingLoader(
                    file_path=str(document),
                    export_type=EXPORT_TYPE,
                    chunker=HybridChunker(tokenizer=tokenizer)
                )
                content = loader.load()
                for chunk in content:
                    chunk.metadata = metadata_simplify(chunk.metadata)
                with open(cached_doc, 'wb') as f:
                    pickle.dump(content, f)
                chunks.extend(content)
            except Exception as e:
                print(f'Error loading {document.name}: {e}')

    end_time = time.time()
    print(f'\nTotal loading time: {end_time - start_time:.2f} seconds')
    return chunks

def ingest():
    chunks = load_documents()
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    db = Path('faiss_db')

    if db.exists():
        print('Loading existing FAISS vector database...')
        vectorstore = FAISS.load_local(
            str(db), 
            embedding,
            allow_dangerous_deserialization=True
        )
    else:
        print('Creating new FAISS vector database...')
        vectorstore = FAISS.from_documents(
            documents=chunks, 
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local(str(db))
    
    return vectorstore

def clip_text(text, threshold=250):
    if len(text) <= threshold:
        return text
    return text[:threshold].rstrip() + '...'

def print_response(resp_dict):
    lines = []
    lines.append(resp_dict['answer'])
    # for i, doc in enumerate(resp_dict['context']):
    #     lines.append(f'\n[Source {i + 1}]')
    #     content = doc.page_content.replace('\n', ' ').strip()
    #     lines.append(f'Text preview: {clip_text(content, threshold=100)}')
    #     page = doc.metadata.get('page', 'N/A')
    #     source = doc.metadata.get('source', 'N/A')
    #     lines.append(f'Page: {page} | Source: {source}')      
    final_string = '\n'.join(lines)
    return final_string
    

ragchain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain

    vectorstore = ingest()
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': TOP_K}
    )
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=GROQ_API_KEY
        )
        question_answer_chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        print('\nAn error occurred while connecting:')  
        print(e)
    yield

app = FastAPI(lifespan=lifespan)

class RequestModel(BaseModel):
    question: str

@app.post('/ask')
def on_request(req: RequestModel):
    question = req.question
    try:
        start_time = time.time()
        resp_dict = rag_chain.invoke({'input': question})
        print(f'Response time: {time.time() - start_time:.2f}s')
        return {'answer': print_response(resp_dict)}
    except Exception as e:
        print('\nAn error occurred while connecting:')  
        print(e)