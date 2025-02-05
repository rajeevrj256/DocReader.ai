from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
import hashlib
from pydantic_ai import RunContext
# Import all the message part classes

from crewAI import trigger_crawler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from agentic_AI_Agent import pydantic_ai_expert, PydanticAIDeps,model,system_prompt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          

def update_env_file(key_name, key_value):
    """Update the .env file with a new API key and reload it."""
    env_file = ".env"

    # Read existing environment variables
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, "r") as file:
            for line in file:
                if "=" in line:
                    var, value = line.strip().split("=", 1)
                    env_vars[var] = value

    # Update or add the new key
    env_vars[key_name] = key_value

    # Write back to .env file
    with open(env_file, "w") as file:
        for var, value in env_vars.items():
            file.write(f"{var}={value}\n")

    # Reload environment variables
    load_dotenv(override=True)

async def hash_domain_name(domain: str) -> str:
    # Create a hash object using SHA-256
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the domain string (encoded to bytes)
    sha256_hash.update(domain.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    return sha256_hash.hexdigest()

async def run_agent_with_streaming(user_input: str,key: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    hashed_key=await hash_domain_name(key)
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        filter_key=hashed_key
    )
    ctx = RunContext(
        model=model,
        usage="query_docs",
        prompt=system_prompt,
        deps=deps,
        
    )

    # Run the agent in a stream
    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
        
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    
    if 'key' not in st.session_state:
        st.session_state.key = ""
    with st.sidebar:
        
        st.header("Trained Me")
        url = st.text_input("Enter URL")
        learn_button = st.button("Learn")
        st.header("Lets start")
        url1=st.text_input("Enter Framework URL")
        start_button = st.button("Start")
        new_api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        if new_api_key:
            # Store the API key in session state
            st.session_state["open_ai_api_key"] = new_api_key
        
        # Update the .env file dynamically
            update_env_file("OPENAI_API_KEY", new_api_key)
        
    
    if learn_button:
        if url :
            print(url)
            st.session_state.key = url 
            api_key = os.getenv("OPENAI_API_KEY")
            print("YOU AIUR",api_key)
            await trigger_crawler(url)
        else:
            st.warning("Please provide both URL")
            
    if start_button:
        if url1 :
            print(url1)
            st.session_state.key = url1 
            if  new_api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                st.warning("Please provide OpenAI API Key")
            
    
    st.markdown(
        """
        <style>
    .logo {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 1000;
    }
    .logo-text {
        font-size: 44px;
        font-weight: 600;
        color: white;
        margin-top: 50px;  /* Adjust the space below the logo */
    }
    </style>
    <div class="logo">
       
    </div>
    <div class="logo-text">
        🗃️DocReader.ai
    </div>
        """, 
        unsafe_allow_html=True
    )
    st.title("Your AI Agentic RAG ")
    st.write("Ask any question about any framework, the hidden truths of the beauty of this framework lie within.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about your framework?")

    if user_input:
        # We append a new request to the conversation explicitly
        key = st.session_state.key
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input+" provide information from this framework of url "+key)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            key = st.session_state.key
        
            print(key)
            if key:  # Ensure key is not empty before running
                await run_agent_with_streaming(user_input, key)
            else:
                st.warning("Please provide a valid URL to proceed.")


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # Fix for Windows subprocesses
    asyncio.run(main())  #