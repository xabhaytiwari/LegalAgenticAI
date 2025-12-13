# Agentic Legal Assistant

An intelligent, multi-role AI agent designed to assist with the Indian Legal System (BNS, IPC, CrPC). Built using LangGraph, this agent moves beyond simple chatbots by autonomously managing state, executing tools, and facilitating complex workflows between Complainers, Police Inspectors, and Prosecutors.

## Key Features

Multi-Role Personality: The agent dynamically adapts its persona and capabilities based on the user:

Citizens (Complainers): Helps draft incident details, files complaints, and checks status.

Inspectors: Accesses case files, submits investigation reports, and queries legal codes.

Commissioners: Assigns cases to officers and oversees department activity.

Prosecutors: Reviews evidence and records final legal decisions.

Agentic Workflow (LangGraph): Uses a graph-based architecture ("Nodes" and "Edges") to decide loop steps—whether to answer directly, search the database, or execute a transaction.

RAG (Retrieval-Augmented Generation): Grounds all legal advice in actual documents (BNS, IPC, CrPC) using a FAISS vector store and HuggingFace embeddings (BAAI/bge-small-en).

Tool Execution: Integrated with Firebase Firestore to perform real-world actions like file_new_complaint, assign_case, and close_case.

Streaming API: Built on FastAPI to support real-time token streaming for a responsive UI.

## Tech Stack

Frameworks: LangChain, LangGraph, FastAPI

LLM: Google Gemini 1.5 Flash (via ChatGoogleGenerativeAI).

Vector Database: FAISS (Facebook AI Similarity Search).

Embeddings: HuggingFace (BAAI/bge-small-en-v1.5).

Database: Firebase Firestore (Admin SDK).

Language: Python 3.10+

## Project Structure

agent.py: The "Brain" - Defines the LangGraph nodes, edges, and state logic.

ingest.py: The ETL Pipeline - Loads PDFs, splits text, creates FAISS index.

tools.py: The "Hands" - Custom functions (Firebase Ops & RAG Search).

main.py: The Interface - FastAPI endpoint for chat streaming.

legal_docs/: Directory for source PDFs (BNS, IPC, Judgments).

## Setup & Installation

Clone the Repository git clone https://github.com/yourusername/agentic-legal-assistant.git cd agentic-legal-assistant

Install Dependencies pip install -r requirements.txt

Environment Variables Create a .env file or export the following: export GOOGLE_API_KEY="your_gemini_api_key"

Database Setup

Place your Firebase Service Account JSON key in the root directory.

Update the filename in tools.py (SERVICE_ACCOUNT_KEY).

Ingest Legal Knowledge Place your legal PDFs in the legal_docs/ folder and run: python ingest.py This creates the bns_vector_store locally.

## Usage

Start the API Server: python main.py The server will start at http://127.0.0.1:8000.

Test with cURL: curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"message": "I want to file a theft complaint", "role": "Complainer"}'

## How it Works (The Graph)

Input: User sends a message + Role.

Call Model (Node): The LLM analyzes the state. It decides if it needs to use a tool (e.g., search_legal_knowledge_base or file_new_complaint).

Decision (Edge):

If Tool Call -> Route to call_tool.

If Response -> Return answer to user.

Call Tool (Node): Executes the Python function (interacting with Firebase or FAISS) and returns the output to the model.

Loop: The model receives the tool output and generates a natural language response.

## Future Improvements

Voice Interface: Adding Speech-to-Text for accessibility in rural areas.

Multilingual Support: Fine-tuning embeddings for Hindi and regional languages.

Frontend: Developing a React/Next.js dashboard for visual case management.