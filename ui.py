
import gradio as gr
import os
import time
import arxiv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings

os.environ["GROQ_API_KEY"] = "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"


def process_papers(query, question_text):
    dirpath = "arxiv_papers"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_order=arxiv.SortOrder.Descending
    )
    
    for result in client.results(search):
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                print(result)
                print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred:", e)
                time.sleep(5)
    
    papers = []
    loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
    full_text = ''
    for paper in papers:
        full_text += paper.page_content
    
    full_text = " ".join(line for line in full_text.splitlines() if line)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])
    
    
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path="./tmp/local_qdrant",
        collection_name="arxiv_papers",
    )
    
    retriever = qdrant.as_retriever()
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_retries=2,
    )
    
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    
    class Question(BaseModel):
        _root_: str
    
    chain = chain.with_types(input_type=Question)
    result = chain.invoke(question_text)
    return result

iface = gr.Interface(
    fn=process_papers,
    inputs=["text", "text"],
    outputs="text",
    description="Enter a search query and a question to process arXiv papers."
)

iface.launch()
