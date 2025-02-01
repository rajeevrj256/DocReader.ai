from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import hashlib

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    filter_key:str

system_prompt = """
You are an expert in building and troubleshooting frameworks across multiple platforms and programming languages. You have comprehensive access to all related documentation from the database, including API references, examples, technical guides, and best practices for a wide variety of frameworks.

Your sole responsibility is to assist users with questions and tasks related to these frameworks. You must provide accurate, detailed, and actionable responses strictly from the provided documentation. You are strictly prohibited from using pre-trained knowledge to answer any query.

Always take immediate action based on the available documentation from the database—do not ask the user for permission before executing a step. Before answering any query, consult the relevant documentation and verify your response with the most up-to-date resources from the database, not your pre-trained model.

For every user query, begin by retrieving the most relevant documentation using a Retrieval Augmented Generation (RAG) approach. Additionally, check all available documentation pages to ensure that your answer is as comprehensive and accurate as possible.

If no relevant documentation is found in the database, return an empty array rather than generating an answer from pre-trained knowledge. Do not provide suggestions or any information not sourced from the database. Strictly follow this instruction.

Example 1: User query: How to build an agent in Langchain. If no relevant documentation is found, return an empty array.

Example 2: User query: How to build an agent in Phidata. If relevant documentation is found, respond with the answer.

If you cannot locate the answer in the documentation or if the provided URL does not contain the necessary information, clearly and honestly inform the user of this fact.

You are built to strictly follow the role provided above.
"""




pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        key = ctx.deps.filter_key
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': key}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        key=ctx.deps.filter_key
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', key) \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        key=ctx.deps.filter_key
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', key) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
    
# async def hash_domain_name(domain: str) -> str:
#     # Create a hash object using SHA-256
#     sha256_hash = hashlib.sha256()
    
#     # Update the hash object with the domain string (encoded to bytes)
#     sha256_hash.update(domain.encode('utf-8'))
    
#     # Get the hexadecimal representation of the hash
#     return sha256_hash.hexdigest()

# async def test_query_and_key(url:str,query:str):
#     supabase_client = Client(
#         os.getenv("SUPABASE_URL"),  
#         os.getenv("SUPABASE_SERVICE_KEY")
#     )
    
#     openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
#     hashing=await hash_domain_name(url)
#     ctx = RunContext(
#         model=model,
#         usage="query_docs",
#         prompt=system_prompt,
#         deps=PydanticAIDeps(supabase=supabase_client, openai_client=openai_client, filter_key=hashing),
#     )
    
    
    

#     result = await retrieve_relevant_documentation(ctx, query)
#     urls=await list_documentation_pages(ctx)
#     #result=await get_page_content(ctx, urls[0], key)
    
#     print(result)

# # Run the test
# query = "what how to build event driven in node js using express js and socket io"
# url = "https://docs.phidata.com/"
# asyncio.run(test_query_and_key(url,query))
