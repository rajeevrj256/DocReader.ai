import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import hashlib

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessChunk:
    url:str
    chunk_number:int
    title:str
    summary:str
    content:str
    metadata:Dict[str, Any]
    embeddings:List[float]
    

#step1: divided the text into chunks according to \n , code block, sentence end,para end
def chunk_text(text:str, chunk_size:int=1000)->List[str]:
    chunks=[]
    start=0
    text_length=len(text)
    
    while start < text_length:
        end=start+chunk_size
         
        if end>=text_length:
            chunks.append(text[start:].strip())
            break
        
        chunk=text[start:end]
        code_block=chunk.rfind('```')
        if code_block!=-1 and code_block>chunk_size*0.3:
            end=start+code_block
        elif '\n\n' in chunk:
            last_break=chunk.rfind('\n\n')
            if last_break>chunk_size*0.3:
                end=start+last_break
        elif '.' in chunk:
            last_period=chunk.rfind('.')
            if last_period>chunk_size*0.3:
                end=start+last_period+1
                
        chunk=text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start=max(start+1,end)
    return chunks



#step2:get organized with tittle and summary on content.
async def get_title_summary(chunk:str,url:str)->Dict[str, str]:
    system_prompt="""
          You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative.
    """
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embeddings(chunk:str)->List[float]:
    try:
        response=await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return [0]*1536
    
async def hash_domain_name(domain: str) -> str:
    # Create a hash object using SHA-256
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the domain string (encoded to bytes)
    sha256_hash.update(domain.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    return sha256_hash.hexdigest()

async def process_chunk(chunk:str,chunk_number:int,url:str,hashing:str)->ProcessChunk:
    extracted=await get_title_summary(chunk,url)
    embeddings=await get_embeddings(chunk)
   
    
    metadata={
        "source":hashing,
        "chunk_size":len(chunk),
        "crawled_at":datetime.now(timezone.utc).isoformat(),
        "url_path":urlparse(url).path
    }
    
    return ProcessChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embeddings=embeddings
    )
    
async def insert_chunks(chunk:ProcessChunk):
    try:
        data={
            "url":chunk.url,
            "chunk_number":chunk.chunk_number,
            "title":chunk.title,
            "summary":chunk.summary,
            "content":chunk.content,
            "metadata":chunk.metadata,
            "embedding":chunk.embeddings
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for url {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk {chunk.chunk_number} for url {chunk.url}: {e}")
        return None
    
async def process_store_doc(url:str,markdown:str,hashing:str):
    chunks=chunk_text(markdown)
    
    tasks=[
        process_chunk(chunk,i,url,hashing) for i,chunk in enumerate(chunks)
    ]
    processed_chunk=await asyncio.gather(*tasks)
    
    insert_tasks=[
        insert_chunks(chunk) for chunk in processed_chunk
    ]
    await asyncio.gather(*insert_tasks)
    

async def crawl_parallel(urls:List[str],hashing:str,max_concurrency:int=5):
    browser_config=BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu","--disable-dev-shm-usage","--no-sandbox"]
    )
    crawl_config=CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
        )
    crawler=AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        seamphore=asyncio.Semaphore(max_concurrency)
        
        async def process_url(url:str):
            async with seamphore:
                result=await crawler.arun(url=url,config=crawl_config,session_id="session1")
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_store_doc(url,result.markdown_v2.raw_markdown,hashing)
                    
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()
        

def get_urls(url:str)->List[str]:
    sitemap_url=f"{url}/sitemap.xml"
    try:
        response=requests.get(sitemap_url)
        response.raise_for_status() # This will raise an exception if the request fails. None.raise_for_status()
        
        root=ElementTree.fromstring(response.content)
        namspaces={"ns":"http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls=[loc.text for loc in root.findall(".//ns:loc",namspaces)]
        return urls
    except Exception as e:
        print(f"Error getting sitemap: {e}")
        return []
    
async def trigger_crawler(url:str):
    
    urls=get_urls(url)
    
    if not urls:
        print("No sitemap found")
        return
    hashing=await hash_domain_name(url)
    await crawl_parallel(urls,hashing)
    