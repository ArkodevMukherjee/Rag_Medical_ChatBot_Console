from langchain_pinecone import PineconeVectorStore # This is the LangChain Pinecone vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage,SystemMessage
import asyncio
import sys
import os

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCxjoaEpXuLqK5TsRB7MG1k8dCA2XJuZe0"

import os

from pinecone import Pinecone

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = "pcsk_4v8jQZ_QBcw4xeV55bkmUUFzVbMQxLrfjrrhrT6EJDrziMaYzfBF1cuGbtFA1VkZMH9N3j"

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

# Initialize the Pinecone client (aliased to avoid conflict)
# PINE_API_KEY = "pcsk_4v8jQZ_QBcw4xeV55bkmUUFzVbMQxLrfjrrhrT6EJDrziMaYzfBF1cuGbtFA1VkZMH9N3j"


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("google_api_key")) # No temperature here for embeddings




# loader = PyPDFLoader("Medical_book.pdf")
# document = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = text_splitter.split_documents(document)

# print(len(docs))

# print(docs)

# Correct way to initialize and populate the Pinecone vector store with documents

from pinecone import ServerlessSpec

index_name = "prac"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)
# batch_size = 100
# num_docs = len(docs)

# for i in range(0, num_docs, batch_size):
#     batch = docs[i:i + batch_size]
#     try:
#         print(f"Upserting batch {i//batch_size + 1}/{(num_docs + batch_size - 1)//batch_size} (documents {i} to {min(i + batch_size, num_docs) - 1})...")
#         vector_store.add_documents(batch)
#         print(f"Batch {i//batch_size + 1} successfully upserted.")
#     except Exception as e:
#         print(f"Error upserting batch {i//batch_size + 1}: {e}")
#         print("This batch failed. Consider reducing 'batch_size' further.")
#         sys.exit(1) # Stop on the first batch error to debug


# vector_store.add_documents(docs)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

async def rag_answer(question: str, vectorstore):
    # 1. Retrieve top 3 relevant documents
    docs = vectorstore.similarity_search(question, k=3)

    # 2. Combine the documents as context
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Format messages
    messages = [
        SystemMessage(content=f"You are a helpful medical tutor. Use the following context to answer questions:\n\n{context}"),
        HumanMessage(content=question)
    ]

    # 4. Call Gemini asynchronously
    response = await llm.apredict_messages(messages)
    return response.content

async def main():
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        answer = await rag_answer(q, vector_store)
        print("\nGemini:", answer)

asyncio.run(main())

# print("Documents successfully indexed into Pinecone index")

