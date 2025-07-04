import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


csv_file = "cars.csv"
df = pd.read_csv(csv_file)


embedding_model = OllamaEmbeddings(model='mxbai-embed-large')

def get_vectorstore():
    if os.path.exists(db_directory) and len(os.listdir(db_directory)) > 0:
        return Chroma(persist_directory=db_directory, embedding_function=embedding_model)

db_directory = "./chrome_langchain_db"
vectorstore_exists = os.path.exists(db_directory) and len(os.listdir(db_directory)) > 0

if not vectorstore_exists:
    documents = []
    ids = []

    
    for idx, row in df.iterrows():
        content = (
            f"{row['YEAR']} {row['Make']} {row['Model']} - Type: {row['TYPE']}, Size: {row['Size']}. "
            f"Power: {row['(kW)']} kW. Rating: {row['RATING']}/10. "
            f"Efficiency: {row['COMB (kWh/100 km)']} kWh/100km combined, {row['CITY (kWh/100 km)']} city, {row['HWY (kWh/100 km)']} highway. "
            f"CO2 Emissions: {row['(g/km)']} g/km. "
            f"Estimated Range: {row['(km)']} km. Charging Time: {row['TIME (h)']} hours."
        )

        documents.append(Document(page_content=content))
        ids.append(f"car-{idx}")


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    splits_docs = text_splitter.split_documents(documents)

    
    question = "What is the most efficient car?"
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        ids=ids,
        persist_directory=db_directory,
    )
    print("New vector store created and saved.")
else:
    
    vectorstore = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model
    )
    print("📦 Existing vector store loaded.")

def get_agent_response(question:str) -> str:
    vectorstore = get_vectorstore()
    retrieved_docs = vectorstore.similarity_search(question, k=3)

    context = "\n".join(doc.page_content for doc in retrieved_docs)

    return context