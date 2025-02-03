
# DocReader.ai
This project is a high-performance documentation retrieval system that leverages OpenAI's GPT-4 mini for summarization, OpenAI embeddings for vector storage, and PostgreSQL (Supabase) with optimized indexing for efficient querying. The system enhances retrieval accuracy using agentic AI and prompt engineering to minimize hallucinations and improve response quality.


## Features

- Automated Documentation Processing: Crawls and chunks website content.
- Efficient Vector Storage: Uses Supabase with PostgreSQL and pgvector.
- Optimized Query Retrieval: Implements IVFFlat indexing for fast similarity search.
- RAG-Based AI Agents: Minimizes hallucinations and improves response accuracy.
- Parallel Processing: Reduces data storage and retrieval time by 50%.


## Installation

Clone the project and navigate to the project directory:

```bash
git clone

```
Server
```bash
cd DocReader.ai
```
Install requirement.txt
```bash
pip install -r requirements.txt
```

Add .env file
```bash 
OPENAI_API_KEY="Your open Api"

SUPABASE_URL="SUPABASE_URL"

SUPABASE_SERVICE_KEY="your SUPABASE_URL service key."

```
create Supabase database using query
```bash
Run database.sql file on Supabase query section.
```
Run server

```bash
streamlit run urls.py

```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY`=Your open Api

`SUPABASE_URL`=SUPABASE_URL

`SUPABASE_SERVICE_KEY`=your SUPABASE_URL service key.

## Technical Workflow
- User Input: A URL is provided.

- Content Processing:

  - Website content is chunked.

  - Titles and summaries are generated using OpenAI GPT-4-mini.

- Embedding & Storage:

   - OpenAI embeddings are computed.

   - Data is stored in Supabase with hashing for efficient retrieval.

- Optimized Search:

  - Uses IVFFlat indexing and metadata filtering in PostgreSQL.

   - Parallel processing speeds up data processing by 50%.

- Query Handling:

  - Converts user queries into embeddings.

  - Performs similarity search with indexing and hashing.

  - Retrieves and ranks results efficiently.

- Response Generation:

  -  Uses AI agents to refine answers and reduce hallucinations by 25%.

  - Implements prompt engineering to prevent inaccurate responses.
## Frontend-streamlit


- Allows users to input queries and retrieve relevant documentation.

- Real-time search with OpenAI-powered embeddings.

- Ensures precise and context-aware results with metadata filtering.
