import asyncio
from pydantic_ai import RunContext
from agentic_AI_Agent import pydantic_ai_expert, PydanticAIDeps

# You need to make sure you have the correct context initialized
# Let's assume that you have a RunContext with the necessary deps for Supabase and OpenAI
# Replace these with actual client instances
supabase_client = ...  # Initialize Supabase client
openai_client = ...    # Initialize OpenAI client

async def test_query_and_key():
    ctx = RunContext(
        deps=PydanticAIDeps(supabase=supabase_client, openai_client=openai_client)
    )

    query = "What are the benefits of using Pydantic AI agents?"
    key = "phidata.com"

    result = await retrieve_relevant_documentation(ctx, query, key)
    print(result)

# Run the test
asyncio.run(test_query_and_key())
