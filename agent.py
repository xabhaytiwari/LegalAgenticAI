import os
from typing import List, TypedDict, Annotated, Sequence
import operator
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama # For local LLM
from langchain_google_genai import ChatGoogleGenerativeAI # For API-based LLM
from langchain_huggingface import HuggingFaceEmbeddings # Or GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# Import the tools we defined in tools.py
from tools import search_legal_knowledge_base, check_complaint_status, file_new_complaint, close_complaint, assign_complaint_to_inspector, submit_inspector_report, submit_prosecutor_decision


# --- 1. Load Retriever and Set Up Tools ---

# Load the embeddings model used in ingest.py
# Make sure this matches! (Ingest Waale se!)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("Embeddings model loaded.")

# Load the local vector store
DB_FAISS_PATH = "bns_vector_store"

if not os.path.exists(DB_FAISS_PATH):
    print(f"Error: Vector store not found at '{DB_FAISS_PATH}'.")
    print("Please run ingest.py first to create the vector store.")
    exit()

db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
print("FAISS vector store loaded and retriever created.")

#class RetrieverInput(BaseModel):
#        """Input schema for the search_legal_knowledge_base tool."""
#        query: str = Field(description="The search query for the legal knowledge base")
        

@tool
def search_legal_knowledge_base(query: str):
         """
        Searches the legal knowledge base (BNS, Procedural Guidelines, etc.)
        for information about laws, procedures, or for drafting complaints.
        """
         print(f"--- Executing RAG search with query: {query} ---")
         try:
            docs = retriever.invoke(query)
            # Format the documents into a single string for the LLM
            return "\n\n".join([doc.page_content for doc in docs])
         except Exception as e:
            print(f"Error during retriever invocation: {e}")
            return f"Error searching knowledge base: {e}"

    # --- END FIX ---

# Create a list of all tools and the tool executor
tools: List[BaseTool] = [search_legal_knowledge_base, check_complaint_status, file_new_complaint, close_complaint, assign_complaint_to_inspector, submit_inspector_report, submit_prosecutor_decision]

# tool_executor = ToolNode(tools)
# Build a simple dictionary to map tool names to their raw functions
tool_map = {tool.name: tool.func for tool in tools}


# --- 2. Set Up The LLM ---

# Option 1: Local LLM with Ollama
# 1. Install Ollama: https://ollama.com/
# 2. Run: ollama pull llama3.1
# 3. Make sure Ollama is running in your terminal
# try:
#    llm = ChatOllama(model="llama3.1", temperature=0)
#    print("ChatOllama (local) model loaded.")
#except Exception as e:
#    print(f"Could not load ChatOllama: {e}")
#    print("Please ensure Ollama is installed, running, and you have pulled a model (e.g., `ollama pull llama3.1`)")
#    llm = None # Will fail later if not set
if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY not set. API calls will fail.")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("ChatGoogleGenerativeAI model loaded.")

if llm is None:
    print("CRITICAL: LLM is not initialized. Please check your setup (Ollama or API key). Exiting.")
    exit()


# --- 3. Define Agent State ---
# This is the memory of the agent.

class AgentState(TypedDict):
    # A list of all messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- 4. Define Agent Nodes ---

def create_agent_prompt():
    """Creates the system prompt for the legal agent."""
    system_prompt = """you are a specialized Legal AI Agent for Indian Law - which includes the BNS, IPC, and CrPC.
Your role is to assist users based on their role, which will be provided with their message.

--- USER'S ROLE ---
The user's message will be prefixed with [USER ROLE: <role>].
You MUST tailor your response to this role.
-   **[USER ROLE: Complainer]**: This user needs help filing, understanding status, or getting next steps. Be empathetic and clear.
-   **[USER ROLE: Inspector]**: This user is a professional. Provide legal interpretations, RAG results, and procedural advice.
-   **[USER ROLE: Prosecutor]**: This user needs high-level legal analysis, case summaries, and report interpretations.
-   **[USER ROLE: Commissioner]**: This user has oversight. Provide summaries and status, and assist with actions like assigning or closing cases.

You have access to seven tools:
1.  `search_legal_knowledge_base`: For legal questions.
2.  `check_complaint_status`: To get the status of a complaint.
3.  `file_new_complaint`: To submit a *new* complaint (for Complainers).
4.  `close_complaint`: To close a complaint with a reason (for Complainers, Commissioners, Prosecutors).
5.  `assign_complaint_to_inspector`: To assign a complaint to an inspector (for Commissioners).
6.  `submit_inspector_report`: To submit a final report (for Inspectors).
7.  `submit_prosecutor_decision`: To record a final decision (for Prosecutors).

Your primary capabilities are:
-   **Legal Guidance & Complaint Drafting:** When a user wants to file a complaint, first use `search_legal_knowledge_base` to find the relevant sections of the BNS, IPC, CrPC. Then, help the user draft the `incident_details` clearly. Finally, once they confirm, use `file_new_complaint` to submit it.
-   **Status & Monitoring:** If a user asks for the status of a case, use `check_complaint_status`.

Always be helpful and precise. If you don't know the answer, say so.
If you use a tool, wait for the tool's output before responding to the user.
Based on the conversation history, decide what to do next:
-   Respond directly to the user.
-   Call one of your available tools.
"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt

# Node 1: The "brain" of the agent
def call_model(state: AgentState):
    """Invokes the LLM to decide the next step or generate a response."""
    print("\n--- NODE: call_model ---")
    if llm is None:
        raise ValueError("LLM is not initialized. Please check your setup (Ollama or API key).")
        
    prompt = create_agent_prompt()
    # Bind the tools to the LLM so it knows how to call them
    model_with_tools = llm.bind_tools(tools)
    
    chain = prompt | model_with_tools
    response = chain.invoke(state)
    
    # The model's response is appended to the message history
    return {"messages": [response]}

# Node 2: The "hands" of the agent
def call_tool(state: AgentState):
    """Executes the tool call decided by the LLM."""
    print("\n--- NODE: call_tool ---")
    # The last message from the model is the tool call
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("Error: No tool call found in the last message.")
        return {"messages": [HumanMessage(content="Error: Model did not produce a valid tool call.")]}

    # Execute all tool calls
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"Executing tool: {tool_name} with args: {tool_args}")
        
        # The tool_executor runs the correct function (e.g., file_new_complaint)
        #response = tool_executor.invoke(tool_call)
        
        # We create a ToolMessage to send back to the model
        #tool_messages.append(ToolMessage(content=str(response), tool_call_id=tool_call["id"]))

            # --- START FIX: Manual Tool Dispatch ---
            # Find the correct tool *function* from our map
        if tool_name not in tool_map:
            response = f"Error: Tool '{tool_name}' not found."
        else:
            tool_function = tool_map[tool_name]
            try:
                # Call the raw function, unpacking the args dict as kwargs
                response = tool_function(**tool_args)
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
                response = f"Error: {e}"
    # --- END FIX ---

    # We create a ToolMessage to send back to the model
    tool_messages.append(ToolMessage(content=str(response), tool_call_id=tool_call["id"]))

    # Append the tool's response to the message history
    return {"messages": tool_messages}


# --- 5. Define Graph Logic ---

def should_continue(state: AgentState) -> str:
    """
    Decides the next step.
    - If the model generated tool calls: run the tools.
    - If the model responded: finish the cycle.
    """
    print("\n--- EDGE: should_continue ---")
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # The model wants to use a tool
        print("Decision: Call tool.")
        return "call_tool"
    else:
        # The model responded directly, so we're done
        print("Decision: End cycle.")
        return END

# --- 6. Compile the Graph ---

def create_agent_graph():
    """Builds and compiles the LangGraph agent."""
    
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)

    # Define the entry point
    workflow.set_entry_point("call_model")

    # Define the edges
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tool": "call_tool",
            END: END,
        },
    )
    
    # This edge always goes from the tools back to the model
    # so the model can see the tool's output
    workflow.add_edge("call_tool", "call_model")

    # We add a checkpointer to the compile step.
    # This tells LangGraph to use MemorySaver to store history.
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    print("\n--- LangGraph agent compiled successfully! ---")
    return app

app = create_agent_graph()

# This allows us to run this file as a test
if __name__ == "__main__":
    if llm is None:
        print("LLM not loaded. Exiting test.")
        exit()

    print("\n--- Starting test conversation ---")
    print("Type 'exit' to quit.")

    # A simple conversation loop
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break

        # We must pass the input as a list of messages
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # The 'stream' method gives us intermediate steps
        for output in app.stream(inputs, {"recursion_limit": 10}):
            # 'output' is a dictionary where keys are node names
            for key, value in output.items():
                if key == "call_model" and value["messages"]:
                    last_msg = value["messages"][-1]
                    if last_msg.tool_calls:
                        print(f"Agent: (Calling tools {', '.join([tc['name'] for tc in last_msg.tool_calls])}...)")
                    else:
                        print(f"Agent: {last_msg.content}")
                elif key == "call_tool":
                    # Optional: Print tool outputs for debugging
                    print(f"Tool Output: {value['messages'][-1].content}")
