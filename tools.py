from langchain.tools import tool
from pydantic import BaseModel, Field
import datetime
import os
# --- Firebase Admin Setup ---
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import ArrayUnion, ArrayRemove
# Path to the service account key
SERVICE_ACCOUNT_KEY = "legal-ai-agent-28659-firebase-adminsdk-fbsvc-4b0e5f65b2.json"

db = None
if os.path.exists(SERVICE_ACCOUNT_KEY):
    try:
        # Initialize Firebase
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred)

        # Get a reference to the Firestore database
        db = firestore.client()
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        print("Please ensure '{SERVICE_ACCOUNT_KEY}' is valid and in the correct path.")
else:
    print(f"Warning: '{SERVICE_ACCOUNT_KEY}' not found.")
    print("Database tools ('check_complaint_status', 'file_new_complaint') will fail.")
    print("Please download it from your Firebase project settings.")

# --- 1. RAG Tool ---
# This tool's logic is a placeholder.
# It gets "bound" to the real FAISS retriever in agent.py
class SearchInput(BaseModel):
    query: str = Field(description="The legal query to search the knowledge base for.")
@tool("search_legal_knowledge_base", args_schema=SearchInput)
def search_legal_knowledge_base(query: str) -> str:
    """
    Searches the legal knowledge base (BNS, Procedural Guidelines, etc.)
    for information related to a legal query.
    Use this to answer questions about laws, procedures, or for drafting complaints.
    """
    # This placeholder is overridden in agent.py by:
    # search_legal_knowledge_base.func = lambda query: retriever.invoke(query)
    return "Placeholder: This function is replaced by the RAG retriever in agent.py"


# --- 2. Complaint Status Tool (Now with Firebase) ---

class CheckStatusInput(BaseModel):
    complaint_id: str = Field(description="The unique ID of the complaint to check.")

@tool("check_complaint_status", args_schema=CheckStatusInput)
def check_complaint_status(complaint_id: str) -> str:
    """
    Checks the status of a specific complaint using its unique ID.
    Returns the current status and details.
    """
    if db is None:
        return "Error: Database connection is not established. Check service key."
    try:
        # Get the document from the 'complaints' collection
        doc_ref = db.collection("complaints").document(complaint_id)
        doc = doc_ref.get()

        if not doc.exists:
            return f"Error: No complaint found with ID: {complaint_id}"
        
        # Format the data into a string response
        data = doc.to_dict()
        status = data.get("status", "N/A")
        details = data.get("incident_details", "N/A")
        filed_on = data.get("filed_on", "N/A")
        
        return f"Status for {complaint_id}: {status}. Filed on: {filed_on}. Details: {details}"

    except Exception as e:
        print(f"Error in check_complaint_status: {e}")
        return f"Error checking status for {complaint_id}: {str(e)}"


# --- 3. File Complaint Tool (Now with Firebase) ---

#class FileComplaintInput(BaseModel):
#    complainant_name: str = Field(description="The full name of the person filing the complaint.")
#    contact_info: str = Field(description="Contact details (phone or email) of the complainant.")
#    incident_details: str = Field(description="A detailed description of the incident, including relevant BNS sections if known.")

# In your tools.py file

# ... (all other code)

# --- 3. File Complaint Tool (Now with Firebase) ---

@tool
def file_new_complaint(
    complainant_name: str, 
    contact_info: str, 
    incident_details: str, 
    submittedBy: str,
    title: str,              # 👈 ADD THIS
    description: str,        # 👈 ADD THIS
    evidenceUrl: str         # 👈 ADD THIS
) -> str:
    """
    Files a new complaint into the system.
    This should only be called after gathering all necessary details from the user.
    Returns a new unique complaint ID.
    
    Args:
        complainant_name: The full name of the person filing the complaint.
        contact_info: Contact details (phone or email) of the complainant.
        incident_details: A detailed description of the incident (for agent context).
        submittedBy: The unique ID of the user filing the complaint.
        title: The title of the complaint.
        description: The full description of the complaint.
        evidenceUrl: The URL to any uploaded evidence.
    """
    if db is None:
        return "Error: Database connection is not established. Check service key."
    try:
        # Create a new complaint document
        data = {
            "complainant_name": complainant_name,
            "contact_info": contact_info,
            "incident_details": incident_details, # We save this for logging/context
            "submittedBy": submittedBy,
            "title": title,              # 👈 ADD THIS
            "description": description,  # 👈 ADD THIS
            "evidenceUrl": evidenceUrl,  # 👈 ADD THIS
            "status": "Filed",
            "filed_on": datetime.datetime.now(datetime.timezone.utc)
        }
        
        # Add a new document with an auto-generated ID
        update_time, doc_ref = db.collection("complaints").add(data)
        
        print(f"New complaint filed with ID: {doc_ref.id} by user: {submittedBy}")
        return f"Complaint filed successfully. Your new complaint ID is: {doc_ref.id}"

    except Exception as e:
        print(f"Error in file_new_complaint: {e}")
        return f"Error filing complaint: {str(e)}"


# --- 4. Close Complaint Tool ---

class CloseComplaintInput(BaseModel):
    complaint_id: str = Field(description="The unique ID of the complaint to close.")
    reason: str = Field(description="A brief reason for closing the complaint.")

@tool("close_complaint", args_schema=CloseComplaintInput)
def close_complaint(complaint_id: str, reason: str) -> str:
    """
    Closes a specific complaint and records the reason.
    This sets the status to 'Closed' and adds a history entry.
    """
    if db is None:
        return "Error: Database connection is not established. Check service key."
    try:
        doc_ref = db.collection("complaints").document(complaint_id)
        doc = doc_ref.get()

        if not doc.exists:
            return f"Error: No complaint found with ID: {complaint_id}"
        
        # Create a new history entry for this action
        new_history_entry = {
            "status": "Closed",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "actor": "AI Legal Agent", # The agent is the actor
            "notes": reason
        }

        # Update the document to set status, add reason, and append history
        doc_ref.update({
            "status": "Closed",
            "closure_reason": reason,
            "history": ArrayUnion([new_history_entry])
        })
        
        print(f"Complaint {complaint_id} closed by agent. Reason: {reason}")
        return f"Complaint {complaint_id} has been successfully closed. Reason: {reason}"

    except Exception as e:
        print(f"Error in close_complaint: {e}")
        return f"Error closing complaint {complaint_id}: {str(e)}"


# --- 5. Assign Complaint Tool (For Commissioner) ---

class AssignComplaintInput(BaseModel):
    complaint_id: str = Field(description="The unique ID of the complaint to assign.")
    inspector_id: str = Field(description="The unique user ID of the inspector.")
    inspector_name: str = Field(description="The full name of the inspector.")

@tool("assign_complaint_to_inspector", args_schema=AssignComplaintInput)
def assign_complaint_to_inspector(complaint_id: str, inspector_id: str, inspector_name: str) -> str:
    """
    Assigns a 'Submitted' complaint to an inspector.
    This sets the status to 'AssignedToInspector' and records who was assigned.
    """
    if db is None:
        return "Error: Database connection is not established."
    try:
        doc_ref = db.collection("complaints").document(complaint_id)
        doc = doc_ref.get()
        if not doc.exists:
            return f"Error: No complaint found with ID: {complaint_id}"
        
        # Create a new history entry
        new_history_entry = {
            "status": "AssignedToInspector",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "actor": "AI Legal Agent (on behalf of Commissioner)",
            "notes": f"Assigned to Inspector: {inspector_name} (ID: {inspector_id})"
        }

        # Update the document
        doc_ref.update({
            "status": "AssignedToInspector",
            "assignedTo": inspector_id,
            "history": ArrayUnion([new_history_entry])
        })
        
        return f"Complaint {complaint_id} successfully assigned to {inspector_name}."

    except Exception as e:
        print(f"Error in assign_complaint_to_inspector: {e}")
        return f"Error assigning complaint: {str(e)}"

# --- 6. Submit Inspector Report Tool (For Inspector) ---

class SubmitReportInput(BaseModel):
    complaint_id: str = Field(description="The unique ID of the complaint.")
    investigation_notes: str = Field(description="The inspector's private investigation notes.")
    final_report: str = Field(description="The inspector's final, conclusive report.")

@tool("submit_inspector_report", args_schema=SubmitReportInput)
def submit_inspector_report(complaint_id: str, investigation_notes: str, final_report: str) -> str:
    """
    Submits the inspector's final report for a complaint.
    This sets the status to 'ReportSubmitted'.
    """
    if db is None:
        return "Error: Database connection is not established."
    try:
        doc_ref = db.collection("complaints").document(complaint_id)
        doc = doc_ref.get()
        if not doc.exists:
            return f"Error: No complaint found with ID: {complaint_id}"
        
        new_history_entry = {
            "status": "ReportSubmitted",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "actor": "AI Legal Agent (on behalf of Inspector)",
            "notes": "Inspector filed final report."
        }

        doc_ref.update({
            "status": "ReportSubmitted",
            "investigationNotes": investigation_notes,
            "inspectorReport": final_report,
            "history": ArrayUnion([new_history_entry])
        })
        
        return f"Final report for {complaint_id} submitted successfully."

    except Exception as e:
        print(f"Error in submit_inspector_report: {e}")
        return f"Error submitting report: {str(e)}"

# --- 7. Submit Prosecutor Decision Tool (For Prosecutor) ---

class SubmitDecisionInput(BaseModel):
    complaint_id: str = Field(description="The unique ID of the complaint.")
    decision: str = Field(description="The prosecutor's final decision or action taken.")

@tool("submit_prosecutor_decision", args_schema=SubmitDecisionInput)
def submit_prosecutor_decision(complaint_id: str, decision: str) -> str:
    """
    Submits the prosecutor's final decision on a complaint.
    This sets the status to 'ActionTaken'.
    """
    if db is None:
        return "Error: Database connection is not established."
    try:
        doc_ref = db.collection("complaints").document(complaint_id)
        doc = doc_ref.get()
        if not doc.exists:
            return f"Error: No complaint found with ID: {complaint_id}"
        
        new_history_entry = {
            "status": "ActionTaken",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "actor": "AI Legal Agent (on behalf of Prosecutor)",
            "notes": "Prosecutor has submitted their decision."
        }

        doc_ref.update({
            "status": "ActionTaken",
            "prosecutorDecision": decision,
            "history": ArrayUnion([new_history_entry])
        })
        
        return f"Prosecutor's decision for {complaint_id} recorded successfully."

    except Exception as e:
        print(f"Error in submit_prosecutor_decision: {e}")
        return f"Error submitting decision: {str(e)}"