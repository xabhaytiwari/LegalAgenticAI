import uvicorn
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage

# Import the compiled LangGraph app from agent.py
try:
    from agent import app as agent_app
    print("Successfully imported agent graph.")
except ImportError:
    print("Error: Could not import 'app' from agent.py.")
    print("Please make sure agent.py exists and is in the same directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

#  FastAPI App
app = FastAPI(
    title="Legal AI Agent API",
    description="API for the Multi-Role Legal AI Agent.",
    version="1.0.0"
)

# Origins that are allowed to make a request
origins = [
    "http://localhost",
    "http://localhost:3000/"
]

#  Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Define API Request/Response Models ---

class ChatRequest(BaseModel):
    """Pydantic model for the incoming chat request."""
    message: str
    conversation_id: Optional[str] = None
    role: str

class ChatResponse(BaseModel):
    """Pydantic model for the outgoing chat response."""
    response: str
    conversation_id: str
    tool_calls: Optional[List[str]] = None

#  Define the Chat Endpoint (Ab chal raha hai, isko chhedna mat koi)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Main endpoint to interact with the Legal AI Agent.
    It streams the agent's response and returns the final answer.
    """
    try:
        # 1. Get or create a unique conversation ID
        conv_id = request.conversation_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": conv_id}}

        user_message_content = f"[USER ROLE: {request.role}]\n\n{request.message}"

        print(f"\n--- Invoking agent for Conversation ID: {conv_id} ---")
        print(f"User (as {request.role}): {request.message}")

        # 2. Prepare the input for the agent
        inputs = {"messages": [HumanMessage(content=user_message_content)]}

        # 3. Stream the agent's execution
        final_response = ""
        tool_calls_list = []

        for output in agent_app.stream(inputs, config=config, stream_mode="values"): 
            # 'output' is the current state of the AgentState graph
            last_message = output["messages"][-1]
            
            if isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    # Collect tool call information
                    print(f"Agent: (Calling tools {', '.join([tc['name'] for tc in last_message.tool_calls])}...)")
                    for tc in last_message.tool_calls:
                        tool_calls_list.append(tc['name'])
                else:
                    # --- Parse the Gemini response (list) ---
                    content = last_message.content
                    
                    if isinstance(content, str):
                        final_response = content
                    elif isinstance(content, list):
                        # This is the complex Gemini case: [{'type': 'text', 'text': '...'}, ...]
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                final_response = part.get('text', '')
                                break  # Found the text, stop looping
                        if not final_response:
                            final_response = "Agent responded, but no text part was found."
                    else:
                        final_response = "Received an unknown response format from agent."
                    
                    print(f"Agent: {final_response}") # Print the parsed text

        # Handle cases where no final response was generated
        if not final_response and not tool_calls_list:
            final_response = "The agent processed your request but did not provide a final answer."
        elif not final_response and tool_calls_list:
            # This happens if the agent's *last* step was a tool call (e.g., filing complaint)
            # The tool *output* will be in the next turn, but for this turn, we confirm the action.
            final_response = f"Agent action confirmed: {', '.join(tool_calls_list)}"

        # 4. Return the structured response
        return ChatResponse(
            response=final_response,
            conversation_id=conv_id,
            tool_calls=tool_calls_list or None
        )

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. Add a simple root endpoint ---
@app.get("/")
def read_root():
    return {"message": "Legal AI Agent API is running."}

# --- 6. Run the server ---
if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)