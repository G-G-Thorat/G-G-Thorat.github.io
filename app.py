from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import os
from transformers import AutoTokenizer, AutoModel
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
import openai

app = Flask(__name__)

# Function to extract text from index.html
def extract_text_from_html(html_file):
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

# Load and process the website content
website_content = extract_text_from_html('index.html')

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

# Step 4: Split text into smaller chunks
from langchain.text_splitter import CharacterTextSplitter

documents = [Document(page_content=website_content)]

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

# Create a vector store
vector_store = Chroma.from_documents(docs, embeddings)

# Initialize OpenAI LLM
openai_api_key = os.getenv('sk-proj-FGyQimcLGmqbomRzYd5JT3BlbkFJMt8komV9JqdBOP0EOPkN')
llm = OpenAI(api_key=openai_api_key)

# Create a conversational retrieval chain
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    max_tokens_limit=1000,
)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    chat_history = request.json.get("chat_history", [])
    response = retrieval_chain.run(question=user_message, chat_history=chat_history)
    return jsonify({"response": response['output']})

if __name__ == "__main__":
    app.run(debug=True)
