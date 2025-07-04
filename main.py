from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from vector import get_agent_response  

app = FastAPI()

class QueryInput(BaseModel):
    question: str

# Load LLM
model = OllamaLLM(model="tinyllama")

template = """
You are a helpful and knowledgeable AI assistant specialized in car information.
Use the following car data to answer the user's question:

{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.get("/")
def Home():
    return {"message": "Welcome to the Car RAG Agent API"}

@app.post("/query")
def query(input: QueryInput):
    
    context = get_agent_response(input.question)
    
    response = chain.invoke({"context": context, "question": input.question})
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
