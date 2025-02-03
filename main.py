from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from typing import List
from agentic_AI_Agent import test_query_and_key;
# Define FastAPI app
app = FastAPI()

# Define request body model
class QueryRequest(BaseModel):
    url: str
    query: str

# Define the endpoint
@app.post("/test_query_and_key/")
async def test_query_and_key_endpoint(request: QueryRequest):
    try:
        # Call the test_query_and_key function with the parameters from the POST request
        print(f"Received request with URL: {request.url} and Query: {request.query}")
        result =await test_query_and_key(request.url, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)