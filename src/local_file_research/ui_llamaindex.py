"""
Streamlit UI for the local file research system using LlamaIndex.
"""

# Import the logging silencer first to suppress all logging errors
try:
    from .logging_silence import silence_all_logging_errors
except ImportError:
    try:
        from src.local_file_research.logging_silence import silence_all_logging_errors
    except ImportError:
        print("Warning: Could not import logging_silence module")

import streamlit as st
import requests
import json
import os
import time
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Set up logging (console only, no file logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get API URL from environment variable or use default
import os
from config import API_PORT
API_URL = os.environ.get("API_URL", f"http://localhost:{API_PORT}")

# Set page config
st.set_page_config(
    page_title="Local File Research",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "current_page" not in st.session_state:
    st.session_state.current_page = "Research"
if "selected_projects" not in st.session_state:
    st.session_state.selected_projects = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "projects" not in st.session_state:
    st.session_state.projects = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = []

# Initialize authentication state
try:
    from src.local_file_research.auth_ui import initialize_auth_state
    initialize_auth_state()
except ImportError:
    # Fallback if import fails
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "requires_2fa" not in st.session_state:
        st.session_state.requires_2fa = False
    if "temp_token" not in st.session_state:
        st.session_state.temp_token = None
    if "totp_secret" not in st.session_state:
        st.session_state.totp_secret = None
    if "totp_qr_code" not in st.session_state:
        st.session_state.totp_qr_code = None
    if "totp_uri" not in st.session_state:
        st.session_state.totp_uri = None
    if "show_2fa_setup" not in st.session_state:
        st.session_state.show_2fa_setup = False

# Function to make API requests
def api_request(endpoint, method="GET", data=None, files=None, auth_type="bearer"):
    """
    Make API request to the backend.

    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data
        files: Request files
        auth_type: Authentication type (bearer or form)
    """
    # Try to use the auth_ui module's API request function
    try:
        from src.local_file_research.auth_ui import make_api_request
        return make_api_request(endpoint, method, data, files, auth_type)
    except ImportError:
        # Fallback to local implementation
        url = f"{API_URL}{endpoint}"
        headers = {}

        # Add authentication token if available
        if auth_type == "bearer" and "auth_token" in st.session_state:
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                if files:
                    response = requests.post(url, headers=headers, data=data, files=files)
                elif auth_type == "form":
                    # For form-based authentication (login endpoint)
                    response = requests.post(
                        url,
                        data=data,
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                else:
                    headers["Content-Type"] = "application/json"
                    response = requests.post(url, headers=headers, json=data)
            elif method == "PUT":
                headers["Content-Type"] = "application/json"
                response = requests.put(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                logger.error(f"Unsupported method: {method}")
                return {"error": f"Unsupported method: {method}"}

            if response.status_code >= 200 and response.status_code < 300:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {"error": f"API request failed: {response.status_code} - {response.text}"}
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {"error": f"API request error: {e}"}

# Function to load projects
def load_projects():
    """
    Load projects from the API.
    """
    response = api_request("/projects/projects")
    if "error" not in response:
        st.session_state.projects = response
    else:
        st.error(f"Failed to load projects: {response['error']}")

# Function to load documents
def load_documents():
    """
    Load documents from the API.
    """
    response = api_request("/documents/documents")
    if "error" not in response:
        # Store documents in session state
        st.session_state.documents = response

        # Try to get document registry info for each document
        try:
            for i, doc in enumerate(st.session_state.documents):
                doc_id = doc.get("document_id")
                if doc_id:
                    # Get document registry info
                    registry_response = api_request(f"/documents/documents/{doc_id}/registry")
                    if "error" not in registry_response:
                        # Update document with registry info
                        st.session_state.documents[i].update(registry_response)
        except Exception as e:
            st.warning(f"Failed to load document registry info: {str(e)}")
    else:
        st.error(f"Failed to load documents: {response['error']}")

# Function to load vector stores
def load_vector_stores():
    """
    Load vector stores from the API.
    """
    response = api_request("/vector-stores")
    if "error" not in response:
        st.session_state.vector_stores = response
    else:
        st.error(f"Failed to load vector stores: {response['error']}")

# Function to search (modify how results are handled)
def search(query, top_k=5, research_mode="rag", report_mode="normal", context_filter=None, project_filter=None, chain=None, analysis_focus="summarize"):
    """
    Search the index.

    Note: The analysis_focus parameter is currently not used in the backend but kept for future implementation.
    """
    data = {
        "query": query,
        "top_k": top_k,
        "session_id": st.session_state.session_id,
        "research_mode": research_mode,
        "report_mode": report_mode,
        "context_filter": context_filter,
        "project_filter": project_filter,
        "chain": chain,
        "analysis_focus": analysis_focus
    }

    response = api_request("/research", method="POST", data=data)
    if "error" not in response:
        logger.info(f"API Response keys: {list(response.keys())}")

        # --- Store the main report and the detailed findings separately ---
        st.session_state.report = response.get("report", "No report generated.")

        # Get findings from the response, ensure they are properly formatted as dictionaries
        # Try to get findings from either "findings" or "results" field for backward compatibility
        findings = response.get("findings", response.get("results", []))
        if isinstance(findings, list):
            st.session_state.detailed_findings = findings
        else:
            # If findings is not a list, create an empty list
            logger.warning(f"Findings is not a list: {type(findings)}")
            st.session_state.detailed_findings = []

        st.session_state.sources_list = response.get("sources", []) # Store sources list

        logger.info(f"Report length: {len(st.session_state.report)}. Detailed findings count: {len(st.session_state.detailed_findings)}")
        # Return the report string primarily, findings are in session state
        return st.session_state.report
    else:
        st.error(f"Search failed: {response['error']}")
        st.session_state.report = None
        st.session_state.detailed_findings = []
        st.session_state.sources_list = []
        return None

# Function to index documents
def index_documents(project_id=None, context_filter=None):
    """
    Index documents.
    """
    data = {
        "session_id": st.session_state.session_id,
        "project_id": project_id,
        "context_filter": context_filter,
        "root_dir": "storage"  # Use the storage directory instead of current directory
    }

    if project_id:
        response = api_request(f"/projects/{project_id}/index", method="POST", data=data)
    else:
        response = api_request("/documents/documents/index", method="POST", data=data)

    if "error" not in response:
        st.success(f"Indexing successful: {response['message']}")
        return True
    else:
        st.error(f"Indexing failed: {response['error']}")
        return False

# Function to upload document
def upload_document(file, project_id=None):
    """
    Upload document to the API.
    """
    files = {"file": (file.name, file, "application/octet-stream")}
    data = {}

    if project_id:
        data["project_id"] = project_id
        st.info(f"Uploading document to project: {project_id}")

    response = api_request("/documents/documents/upload", method="POST", data=data, files=files)

    if "error" not in response:
        st.success(f"Upload successful: {response['message']}")

        # Add debug info
        st.info(f"Document ID: {response.get('document_id')}")
        st.info(f"Project ID: {project_id}")

        # Reload documents to ensure we have the latest data
        load_documents()

        return True
    else:
        st.error(f"Upload failed: {response['error']}")
        if project_id:
            st.error(f"Failed to upload to project: {project_id}")
        return False

# Function to create project
def create_project(name, description=None):
    """
    Create a new project.
    """
    data = {
        "name": name,
        "description": description or ""
    }

    response = api_request("/projects/projects", method="POST", data=data)

    if "error" not in response:
        st.success(f"Project created: {response['name']}")
        load_projects()  # Reload projects
        return response
    else:
        st.error(f"Project creation failed: {response['error']}")
        return None

# Function to delete project
def delete_project(project_id):
    """
    Delete a project.
    """
    response = api_request(f"/projects/projects/{project_id}", method="DELETE")

    if "error" not in response:
        st.success(f"Project deleted: {response['message']}")
        load_projects()  # Reload projects
        return True
    else:
        st.error(f"Project deletion failed: {response['error']}")
        return False

# Function to delete document
def delete_document(document_id):
    """
    Delete a document.
    """
    response = api_request(f"/documents/documents/{document_id}", method="DELETE")

    if "error" not in response:
        st.success(f"Document deleted: {response['message']}")
        load_documents()  # Reload documents
        return True
    else:
        st.error(f"Document deletion failed: {response['error']}")
        return False

# Function to delete vector store
def delete_vector_store(store_id, store_type):
    """
    Delete a vector store.
    """
    response = api_request(f"/vector-stores/{store_id}?store_type={store_type}", method="DELETE")

    if "error" not in response:
        st.success(f"Vector store deleted: {response['message']}")
        load_vector_stores()  # Reload vector stores
        return True
    else:
        st.error(f"Vector store deletion failed: {response['error']}")
        return False

# Function to migrate to LlamaIndex
def migrate_to_llamaindex():
    """
    Migrate existing vector stores to LlamaIndex format.
    """
    response = api_request("/migrate-to-llamaindex", method="POST")

    if "error" not in response:
        st.success(f"Migration successful: {response['message']}")
        load_vector_stores()  # Reload vector stores
        return True
    else:
        st.error(f"Migration failed: {response['error']}")
        return False

# Sidebar navigation
def sidebar():
    """
    Render sidebar navigation.
    """
    st.sidebar.title("Local File Research")

    # User info
    try:
        from src.local_file_research.auth_ui import user_info_sidebar
        user_info_sidebar()
    except ImportError:
        # Fallback if import fails
        if st.session_state.get("authenticated", False) and st.session_state.get("username"):
            st.sidebar.subheader("User")
            st.sidebar.text(f"Logged in as: {st.session_state.username}")

            # Add security settings link
            if st.sidebar.button("Security Settings (2FA)"):
                st.session_state.current_page = "Security"
                st.rerun()

            # Add logout button
            if st.sidebar.button("Logout"):
                # Clear session state
                st.session_state.authenticated = False
                st.session_state.auth_token = None
                st.session_state.username = None
                # Clear 2FA-related session state
                if "requires_2fa" in st.session_state:
                    st.session_state.requires_2fa = False
                if "temp_token" in st.session_state:
                    st.session_state.temp_token = None
                if "totp_secret" in st.session_state:
                    st.session_state.totp_secret = None
                if "totp_qr_code" in st.session_state:
                    st.session_state.totp_qr_code = None
                if "totp_uri" in st.session_state:
                    st.session_state.totp_uri = None
                if "show_2fa_setup" in st.session_state:
                    st.session_state.show_2fa_setup = False

                st.sidebar.success("Logged out successfully!")
                st.rerun()
        else:
            st.sidebar.subheader("Authentication")
            st.sidebar.markdown("You need to login to access all features.")

            # Create a button to show the auth page
            if st.sidebar.button("Login / Register"):
                st.session_state.current_page = "Auth"
                st.rerun()

            # Set authenticated to false to ensure proper state
            st.session_state.authenticated = False

    # Navigation
    pages = ["Research", "Documents", "Projects", "Vector Stores", "Settings"]

    # Add Security page if logged in
    if st.session_state.get("authenticated", False) and st.session_state.get("auth_token"):
        pages.append("Security")

    # Add Auth page if not logged in
    if "auth_token" not in st.session_state or not st.session_state.auth_token:
        pages = ["Auth"] + pages

    selected_page = st.sidebar.radio("Navigation", pages)

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    # Project selection
    if st.session_state.current_page == "Research":
        st.sidebar.subheader("Project Selection")

        # Load projects if not loaded
        if not st.session_state.projects:
            load_projects()

        # Create project selection
        # Handle different project data structures
        project_names = []
        for p in st.session_state.projects:
            if "name" in p:
                project_names.append(p["name"])
            else:
                # Fallback to project_id if name is not available
                project_names.append(p.get("project_id", "Unknown Project"))

        if not project_names:
            st.sidebar.warning("No projects found. Please create a project first.")
            if st.sidebar.button("Go to Projects Page"):
                st.session_state.current_page = "Projects"
                st.rerun()
        else:
            selected_projects = st.sidebar.multiselect(
                "Select Projects to Search",
                project_names,
                help="Select one or more projects to search within"
            )

            if selected_projects != st.session_state.selected_projects:
                st.session_state.selected_projects = selected_projects

            # Add a note about project selection
            if not selected_projects:
                st.sidebar.warning("⚠️ Please select at least one project to search")
            else:
                st.sidebar.success(f"✅ Searching in {len(selected_projects)} project(s)")

    # Session ID
    st.sidebar.subheader("Session")
    st.sidebar.text(f"Session ID: {st.session_state.session_id}")

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        load_projects()
        load_documents()
        load_vector_stores()
        st.success("Data refreshed")

# Research page (modify how results are displayed)
def research_page():
    """
    Render research page.
    """
    st.title("Research")
    # ... (Keep project selection and form setup the same) ...
    st.markdown("""
    **Research Workflow:**
    1. Select one or more projects from the sidebar
    2. Enter your search query
    3. Choose research and report modes
    4. Click Search to find relevant information
    """)

    # Initialize session state variables for form fields if they don't exist
    if "research_query" not in st.session_state: st.session_state.research_query = ""
    if "research_top_k" not in st.session_state: st.session_state.research_top_k = 5
    if "research_mode" not in st.session_state: st.session_state.research_mode = "rag"
    if "report_mode" not in st.session_state: st.session_state.report_mode = "normal"
    if "context_filter" not in st.session_state: st.session_state.context_filter = ""
    if "agent_chain" not in st.session_state: st.session_state.agent_chain = None
    if "analysis_focus" not in st.session_state: st.session_state.analysis_focus = "summarize" # Default focus

    # Callbacks
    def update_research_mode():
        if st.session_state.research_mode != "multi_iteration": st.session_state.agent_chain = None
    def update_report_mode(): pass # Keep function, logic removed

    st.subheader("Search Parameters")
    query = st.text_area("Search Query", height=100, key="research_query")
    col1, col2, col3 = st.columns(3)
    with col1: top_k = st.number_input("Top K Results", min_value=1, max_value=100, value=st.session_state.research_top_k, key="research_top_k")
    with col2: research_mode = st.selectbox("Research Mode", ["rag", "multi_iteration"], key="research_mode", on_change=update_research_mode)
    with col3: report_format = st.selectbox("Report Format", ["normal", "chain_of_thought", "enhanced"], key="report_mode", on_change=update_report_mode)

    st.subheader("Analysis Options")
    # col4, col5 = st.columns(2)
    # TEMPORARILY DISABLED: analysis_focus dropdown is commented out since it's not currently used in the backend
    # This can be uncommented when the feature is implemented in the future
    # with col4: analysis_focus = st.selectbox("Analysis Focus", ["summarize", "answer", "extract", "analyze"], key="analysis_focus")

    # Changed from col5 to full width since we removed the other column
    context_filter = st.text_input("Context Filter (optional)", key="context_filter")

    # Project filter
    project_filter = st.session_state.selected_projects[0] if st.session_state.selected_projects else None

    # Agent chain selection
    chain = None
    if research_mode == "multi_iteration":
        # Display chain selection below context filter (since we removed columns)
        st.subheader("Agent Chain Selection")
        # (Keep chain selection logic as before)
        chain_categories = { "Analysis Chains": ["None","analyze_document","enhanced_analysis","deep_research","comprehensive_analysis"], "Reasoning Chains": ["cot_then_answer","interpretation_chain","multi_iteration_chain"], "Verification Chains": ["summarize_then_fact_check","verification_chain"], "Specialized Chains": ["proposal_chain","technical_chain"]}
        chain_options = []
        for category, chains in chain_categories.items():
            chain_options.extend(chains)
        chain_help = "💡 For enhanced reporting, 'deep_research' is recommended" if st.session_state.report_mode == "enhanced" else ""
        chain = st.selectbox("Agent Chain", chain_options, key="agent_chain", help=chain_help)
        if chain == "None":
            chain = None


    # Search button
    submitted = st.button("Search")

    if submitted and query:
        if not st.session_state.selected_projects:
             st.warning("⚠️ Please select at least one project from the sidebar before searching.")
        else:
            with st.spinner("Searching..."):
                report_content = search( # search now returns the report string
                    query=st.session_state.research_query,
                    top_k=st.session_state.research_top_k,
                    research_mode=st.session_state.research_mode,
                    report_mode=st.session_state.report_mode,
                    context_filter=st.session_state.context_filter,
                    project_filter=project_filter,
                    chain=st.session_state.agent_chain,
                    # Still passing analysis_focus for API compatibility, though it's not used in the backend
                    analysis_focus="summarize" # Using default value since the dropdown is now hidden
                )
                # Trigger rerun to display results stored in session state
                st.rerun()


    # --- Display results ---
    # Check if report exists in session state
    if 'report' in st.session_state and st.session_state.report:
        results_container = st.container()
        with results_container:
            st.subheader("📝 Research Report")
            # Display the main report content
            st.markdown(st.session_state.report)
            st.markdown("---") # Separator

            # Add an expander for detailed findings
            if 'detailed_findings' in st.session_state and st.session_state.detailed_findings:
                with st.expander(f"View Detailed Findings ({len(st.session_state.detailed_findings)})"):
                    for i, finding in enumerate(st.session_state.detailed_findings):
                        if not isinstance(finding, dict): continue # Skip non-dicts

                        # Display finding details - Reuse formatting helper if needed
                        ref = finding.get('reference_number', i + 1)
                        title = finding.get('source_name', finding.get('file_path', f'Finding {ref}')).split('/')[-1]
                        # Get the normalized score (should be between 0 and 1)
                        score = finding.get('score', 0)

                        # Format the score as a percentage
                        score_str = f"{score:.4f} ({score*100:.1f}%)"

                        st.markdown(f"**Finding {ref}: {title} (Score: {score_str})**")
                        st.markdown(f"**Content Snippet:** {finding.get('content', 'N/A')[:500]}...")

                        # Display summary if available
                        if finding.get('summary'):
                            st.markdown(f"**Summary:** {finding.get('summary')}")

                        # Display analysis as formatted text instead of JSON
                        if finding.get('analysis'):
                            analysis = finding.get('analysis')
                            st.markdown("**Analysis:**")
                            if isinstance(analysis, dict):
                                for key, value in analysis.items():
                                    if value:
                                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                            else:
                                st.markdown(str(analysis))

                        # Display agent chain results if available
                        agent_fields = {
                            "source_type": "Source Type",
                            "chain_status": "Chain Status",
                            "interpretation": "Interpretation",
                            "insights": "Insights",
                            "limitations": "Limitations",
                            "recommendations": "Recommendations",
                            "next_steps": "Next Steps",
                            "alternatives": "Alternatives",
                            "rationale": "Rationale",
                            "details": "Details",
                            "challenges": "Challenges",
                            "solutions": "Solutions"
                        }

                        for field, title in agent_fields.items():
                            if field in finding and finding[field]:
                                value = finding[field]
                                st.markdown(f"**{title}:**")
                                if isinstance(value, list):
                                    for item in value:
                                        st.markdown(f"- {item}")
                                else:
                                    st.markdown(str(value))

                        # Display article information if available
                        if "article" in finding and finding["article"]:
                            st.markdown("**Article:**")
                            st.markdown(f"{finding['article'][:300]}...")

                            # Add article metadata
                            if "article_type" in finding:
                                st.markdown(f"**Article Type:** {finding['article_type']}")
                            if "key_themes" in finding and finding["key_themes"]:
                                st.markdown("**Key Themes:**")
                                themes = finding["key_themes"]
                                # Handle different formats of key_themes
                                if isinstance(themes, list):
                                    for theme in themes:
                                        # Check if theme is a single character (likely from DSPy agent)
                                        if isinstance(theme, str) and len(theme.strip()) <= 1:
                                            continue
                                        st.markdown(f"- {theme}")
                                elif isinstance(themes, str):
                                    # If it's a string, split by newlines or commas
                                    if '\n' in themes:
                                        for line in themes.split('\n'):
                                            line = line.strip()
                                            if line and not line.startswith('-'):
                                                st.markdown(f"- {line}")
                                            elif line:
                                                st.markdown(f"{line}")
                                    else:
                                        for theme in themes.split(','):
                                            theme = theme.strip()
                                            if theme:
                                                st.markdown(f"- {theme}")
                            if "word_count" in finding:
                                st.markdown(f"**Word Count:** {finding['word_count']}")

                        # Add verification information if available
                        verification = finding.get('verification', {})
                        if verification:
                            st.markdown("**Verification:**")
                            for vkey, vvalue in verification.items():
                                st.markdown(f"**{vkey.replace('_', ' ').title()}:** {vvalue}")

                        st.markdown("---")
            else:
                st.info("No detailed findings available for this report.")

            # Add sources list (optional expander)
            if 'sources_list' in st.session_state and st.session_state.sources_list:
                 with st.expander("View Sources"):
                      st.json(st.session_state.sources_list) # Display as JSON for now

        # Add a button to clear results
        if st.button("Clear Results", use_container_width=True):
            st.session_state.report = None
            st.session_state.detailed_findings = []
            st.session_state.sources_list = []
            st.rerun()

    elif submitted and not query:
         st.warning("Please enter a search query.")

# Documents page
def documents_page():
    """
    Render documents page.
    """
    st.title("Documents")

    # Load documents if not loaded
    if not st.session_state.documents:
        load_documents()

    # Information about document management
    st.markdown("""
    ## Document Management

    This page shows all documents in the system. To add new documents:

    1. Go to the **Projects** page
    2. Create a new project or select an existing one
    3. Use the upload form in the project details to add documents
    4. Index the project to make documents searchable

    Documents are organized by projects for better management and context.
    """)

    # Add a button to go to Projects page
    if st.button("Go to Projects Page"):
        st.session_state.current_page = "Projects"
        st.rerun()

    # Add a note about indexing
    st.info("📌 To index documents, please go to the Projects page, create or select a project, and use the 'Index Project' button.")

    # Display documents
    if st.session_state.documents:
        st.subheader(f"Documents ({len(st.session_state.documents)})")

        # Create DataFrame
        data = []
        for doc in st.session_state.documents:
            data.append({
                "ID": doc.get("document_id", "Unknown"),
                "Name": doc.get("name", "Unknown"),
                "Type": doc.get("type", "Unknown"),
                "Size": doc.get("size", 0),
                "Projects": ", ".join(doc.get("projects", [])),
                "Created": doc.get("created", "Unknown")
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

        # Document details
        st.subheader("Document Details")
        selected_doc_id = st.selectbox("Select Document", [doc["document_id"] for doc in st.session_state.documents])

        if selected_doc_id:
            # Find selected document
            selected_doc = None
            for doc in st.session_state.documents:
                if doc["document_id"] == selected_doc_id:
                    selected_doc = doc
                    break

            if selected_doc:
                # Display document details
                st.json(selected_doc)

                # Delete button
                if st.button(f"Delete Document {selected_doc_id}"):
                    with st.spinner("Deleting..."):
                        success = delete_document(selected_doc_id)

                        if success:
                            # Reload documents
                            load_documents()
                            st.rerun()

# Projects page
def projects_page():
    """
    Render projects page.
    """
    st.title("Projects")

    st.markdown("""
    Projects help you organize your documents and research. Each project can contain multiple documents
    that can be indexed together for better context-aware search.

    **Workflow:**
    1. Create a project
    2. Upload documents to the project
    3. Index the project
    4. Search your documents in the Research page
    """)

    # Load projects if not loaded
    if not st.session_state.projects:
        load_projects()

    # Create project form
    with st.form("create_project_form"):
        project_name = st.text_input("Project Name")
        project_description = st.text_area("Project Description")

        submitted = st.form_submit_button("Create Project")

        if submitted and project_name:
            with st.spinner("Creating project..."):
                success = create_project(project_name, project_description)

                if success:
                    # Reload projects
                    load_projects()

    # Display projects
    if st.session_state.projects:
        st.subheader(f"Projects ({len(st.session_state.projects)})")

        # Create DataFrame
        data = []
        for project in st.session_state.projects:
            data.append({
                "ID": project.get("project_id", "Unknown"),
                "Name": project.get("name", project.get("project_id", "Unknown")),
                "Description": project.get("description", ""),
                "Documents": len(project.get("documents", [])),
                "Created": project.get("created", project.get("updated", "Unknown"))
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

        # Project details
        st.subheader("Project Details")
        selected_project_id = st.selectbox("Select Project", [project.get("project_id", "Unknown") for project in st.session_state.projects])

        if selected_project_id:
            # Find selected project
            selected_project = None
            for project in st.session_state.projects:
                if project.get("project_id") == selected_project_id:
                    selected_project = project
                    break

            if selected_project:
                # Display project details
                st.json(selected_project)

                # Get project name
                project_name = selected_project.get("name", selected_project.get("project_id", "Unknown"))

                # Document upload section
                st.subheader(f"Upload Documents to {project_name}")

                with st.form(key=f"upload_to_project_{selected_project_id}"):
                    uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "docx", "md", "csv", "json"])
                    upload_submitted = st.form_submit_button("Upload to Project")

                    if upload_submitted and uploaded_file:
                        with st.spinner("Uploading..."):
                            success = upload_document(uploaded_file, selected_project_id)

                            if success:
                                st.success(f"Document uploaded to project {project_name}")
                                # Reload documents
                                load_documents()

                # Project documents
                st.subheader(f"Documents in {project_name}")

                # Force reload documents to get the latest data
                load_documents()

                # Get documents for this project directly from the API
                response = api_request(f"/documents/documents?project_id={selected_project_id}")

                if "error" not in response:
                    project_docs = response
                    st.info(f"Found {len(project_docs)} documents in project {project_name}")
                else:
                    project_docs = []
                    st.error(f"Failed to load project documents: {response.get('error')}")

                    # Fallback to filtering documents in session state
                    for doc in st.session_state.documents:
                        # Check if project_id matches directly
                        if doc.get("project_id") == selected_project_id:
                            project_docs.append(doc)
                        # Check if project is in the projects list
                        elif "projects" in doc and selected_project_id in doc.get("projects", []):
                            project_docs.append(doc)

                    st.info(f"Found {len(project_docs)} documents in project {project_name} (fallback method)")

                # Debug info
                if not project_docs and st.session_state.documents:
                    st.info("No documents found in this project. Here's a sample document structure:")
                    sample_doc = st.session_state.documents[0]
                    st.json(sample_doc)

                if project_docs:
                    # Create DataFrame
                    data = []
                    for doc in project_docs:
                        data.append({
                            "ID": doc.get("document_id", "Unknown"),
                            "Name": doc.get("title", doc.get("name", "Unknown")),
                            "Type": doc.get("document_type", "Unknown"),
                            "Size": doc.get("size", 0),
                            "Created": doc.get("created_at", "Unknown")
                        })

                    df = pd.DataFrame(data)
                    st.dataframe(df)
                else:
                    st.info("No documents in this project yet. Upload documents using the form above.")

                # Index button
                if st.button(f"Index Project {project_name}"):
                    with st.spinner("Indexing..."):
                        success = index_documents(selected_project_id)

                        if success:
                            # Reload projects
                            load_projects()
                            st.success(f"Project {project_name} indexed successfully!")

                # Delete button
                if st.button(f"Delete Project {project_name}"):
                    with st.spinner("Deleting..."):
                        success = delete_project(selected_project_id)

                        if success:
                            # Reload projects
                            load_projects()
                            st.rerun()

# Vector Stores page
def vector_stores_page():
    """
    Render vector stores page.
    """
    st.title("Vector Stores")

    # Load vector stores if not loaded
    if not st.session_state.vector_stores:
        load_vector_stores()

    # Migration button
    if st.button("Migrate to LlamaIndex"):
        with st.spinner("Migrating..."):
            success = migrate_to_llamaindex()

            if success:
                # Reload vector stores
                load_vector_stores()

    # Display vector stores
    if st.session_state.vector_stores:
        st.subheader(f"Vector Stores ({len(st.session_state.vector_stores)})")

        # Create DataFrame
        data = []
        for store in st.session_state.vector_stores:
            data.append({
                "ID": store.get("project_id", store.get("session_id", "Unknown")),
                "Type": store.get("type", "Unknown"),
                "Documents": store.get("document_count", 0),
                "Chunks": store.get("chunk_count", 0),
                "Last Indexed": store.get("last_indexed", "Unknown")
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

        # Vector store details
        st.subheader("Vector Store Details")
        selected_store_id = st.selectbox("Select Vector Store", [store.get("project_id", store.get("session_id", "Unknown")) for store in st.session_state.vector_stores])

        if selected_store_id:
            # Find selected vector store
            selected_store = None
            for store in st.session_state.vector_stores:
                if store.get("project_id", store.get("session_id", "Unknown")) == selected_store_id:
                    selected_store = store
                    break

            if selected_store:
                # Display vector store details
                st.json(selected_store)

                # Delete button
                if st.button(f"Delete Vector Store {selected_store_id}"):
                    with st.spinner("Deleting..."):
                        success = delete_vector_store(selected_store_id, selected_store.get("type", "session"))

                        if success:
                            # Reload vector stores
                            load_vector_stores()
                            st.rerun()

# Settings page
def settings_page():
    """
    Render settings page.
    """
    st.title("Settings")

    # API URL
    st.subheader("API Settings")
    api_url = st.text_input("API URL", value=API_URL)

    if st.button("Update API URL"):
        # Update API URL
        st.session_state.api_url = api_url
        st.success(f"API URL updated to {api_url}")

    # Session ID
    st.subheader("Session Settings")
    session_id = st.text_input("Session ID", value=st.session_state.session_id)

    if st.button("Update Session ID"):
        # Update session ID
        st.session_state.session_id = session_id
        st.success(f"Session ID updated to {st.session_state.session_id}")

    # Clear cache
    st.subheader("Cache Settings")

    if st.button("Clear Cache"):
        # Clear cache
        st.session_state.search_results = []
        st.session_state.documents = []
        st.session_state.projects = []
        st.session_state.vector_stores = []
        st.success("Cache cleared")

# Main function
def security_page():
    """
    Render security settings page.
    """
    st.title("Security Settings")

    # Check if user is authenticated
    if not st.session_state.get("authenticated", False) or not st.session_state.get("auth_token"):
        st.error("You must be logged in to access security settings")
        return

    # Create tabs for different security settings
    tab1, tab2 = st.tabs(["Two-Factor Authentication", "Account"])

    with tab1:
        setup_2fa_section()

    with tab2:
        st.subheader("Account Information")
        user_info = api_request("/auth/me")
        if "error" not in user_info:
            st.write(f"**Username:** {user_info['username']}")
            st.write(f"**Email:** {user_info['email']}")
            st.write(f"**Role:** {user_info['role']}")

            # Add password change form
            st.subheader("Change Password")
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                submit = st.form_submit_button("Change Password")

                if submit:
                    if not current_password or not new_password:
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("New passwords do not match")
                    else:
                        with st.spinner("Changing password..."):
                            response = api_request(
                                "/auth/change-password",
                                method="POST",
                                data={
                                    "current_password": current_password,
                                    "new_password": new_password
                                }
                            )

                            if "error" not in response:
                                st.success("Password changed successfully")
                            else:
                                st.error(f"Failed to change password: {response['error']}")
        else:
            st.error(f"Failed to get user information: {user_info['error']}")

def setup_2fa_section():
    """
    Render 2FA setup section.
    """
    st.subheader("Two-Factor Authentication (2FA)")

    # Add detailed explanation about 2FA
    st.info("""
    **What is Two-Factor Authentication?**

    Two-factor authentication (2FA) adds an extra layer of security to your account by requiring:
    1. Something you know (your password)
    2. Something you have (a code from your authenticator app)

    This means that even if someone gets your password, they still can't access your account without your authenticator app.

    **How to set up 2FA:**
    1. Click the "Set Up Two-Factor Authentication" button below
    2. Scan the QR code with an authenticator app like Google Authenticator, Microsoft Authenticator, or Authy
    3. Enter the verification code from your app to complete the setup
    """)

    # Add recommended apps
    st.markdown("**Recommended Authenticator Apps:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[Google Authenticator](https://play.google.com/store/apps/details?id=com.google.android.apps.authenticator2)")
    with col2:
        st.markdown("[Microsoft Authenticator](https://www.microsoft.com/en-us/security/mobile-authenticator-app)")
    with col3:
        st.markdown("[Authy](https://authy.com/download/)")

    st.markdown("---")

    # Get user info
    user_info = api_request("/auth/me")
    if "error" in user_info:
        st.error(f"Failed to get user information: {user_info['error']}")
        return

    # Check if 2FA is already enabled
    if user_info.get("totp_enabled", False):
        st.success("Two-factor authentication is already enabled for your account")

        # Disable 2FA option
        with st.form("disable_2fa_form"):
            st.warning("Disabling two-factor authentication will reduce the security of your account")
            password = st.text_input("Enter your password to confirm", type="password")
            submit = st.form_submit_button("Disable 2FA")

            if submit and password:
                with st.spinner("Disabling 2FA..."):
                    response = api_request(
                        "/auth/2fa/disable",
                        method="POST",
                        data={"password": password}
                    )

                    if "error" not in response:
                        st.success("Two-factor authentication has been disabled")
                        st.rerun()
                    else:
                        st.error(f"Failed to disable 2FA: {response['error']}")
    else:
        # Set up 2FA
        if st.button("Set Up Two-Factor Authentication"):
            with st.spinner("Setting up 2FA..."):
                response = api_request("/auth/2fa/setup", method="POST")

                if "error" not in response:
                    # Store setup information in session state
                    st.session_state.totp_secret = response["secret"]
                    st.session_state.totp_qr_code = response["qr_code"]
                    st.session_state.totp_uri = response["uri"]
                    st.session_state.show_2fa_setup = True
                else:
                    st.error(f"Failed to set up 2FA: {response['error']}")

        # Show 2FA setup instructions if available
        if st.session_state.get("show_2fa_setup", False):
            st.success("Two-factor authentication setup initiated")

            # Display QR code
            st.subheader("1. Scan this QR code with your authenticator app")
            qr_code = st.session_state.totp_qr_code
            st.image(f"data:image/png;base64,{qr_code}", width=300)

            # Display manual entry option
            st.subheader("2. Or enter this code manually")
            st.code(st.session_state.totp_secret)

            # Verify and enable 2FA
            st.subheader("3. Verify and enable 2FA")
            with st.form("enable_2fa_form"):
                totp_token = st.text_input("Enter the verification code from your app")
                submit = st.form_submit_button("Verify and Enable")

                if submit and totp_token:
                    with st.spinner("Verifying and enabling 2FA..."):
                        response = api_request(
                            "/auth/2fa/enable",
                            method="POST",
                            data={"totp_token": totp_token}
                        )

                        if "error" not in response:
                            # Clear setup information
                            if "totp_secret" in st.session_state:
                                del st.session_state.totp_secret
                            if "totp_qr_code" in st.session_state:
                                del st.session_state.totp_qr_code
                            if "totp_uri" in st.session_state:
                                del st.session_state.totp_uri
                            if "show_2fa_setup" in st.session_state:
                                del st.session_state.show_2fa_setup

                            st.success("Two-factor authentication has been enabled for your account")
                            st.rerun()
                        else:
                            st.error(f"Failed to enable 2FA: {response['error']}")

def verify_2fa_form():
    """
    Render 2FA verification form.
    """
    st.subheader("Two-Factor Authentication")
    st.info("Please enter the verification code from your authenticator app")

    # 2FA verification form
    with st.form("2fa_form"):
        totp_token = st.text_input("Verification Code")
        submit = st.form_submit_button("Verify")

        if submit and totp_token:
            with st.spinner("Verifying..."):
                try:
                    # Make API request to verify 2FA
                    response = api_request(
                        "/auth/2fa/verify",
                        method="POST",
                        data={
                            "token": st.session_state.temp_token,
                            "totp_token": totp_token
                        }
                    )

                    if "error" not in response:
                        # Store token in session state
                        st.session_state.auth_token = response["access_token"]
                        st.session_state.authenticated = True
                        st.session_state.requires_2fa = False
                        st.session_state.temp_token = None
                        st.success("Two-factor authentication successful!")
                        st.rerun()
                    else:
                        st.error(f"Verification failed: {response['error']}")
                except Exception as e:
                    st.error(f"Verification error: {str(e)}")

def login_form_integrated():
    """
    Render login form with 2FA support.
    """
    # Check if 2FA verification is required
    if st.session_state.get("requires_2fa", False) and st.session_state.get("temp_token"):
        verify_2fa_form()
        return

    st.subheader("Login")

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit and username and password:
            with st.spinner("Logging in..."):
                try:
                    # Make API request to login
                    response = api_request(
                        "/auth/login",
                        method="POST",
                        data={"username": username, "password": password},
                        auth_type="form"
                    )

                    if "error" not in response:
                        # Check if 2FA is required
                        if response.get("requires_2fa", False):
                            # Store temporary token and username
                            st.session_state.temp_token = response["access_token"]
                            st.session_state.username = username
                            st.session_state.requires_2fa = True
                            st.info("Two-factor authentication required")
                            st.rerun()  # Rerun to show 2FA form
                        else:
                            # Normal login without 2FA
                            st.session_state.auth_token = response["access_token"]
                            st.session_state.username = username
                            st.session_state.authenticated = True
                            st.success(f"Welcome, {username}!")
                            st.rerun()
                    else:
                        st.error(f"Login failed: {response['error']}")
                except Exception as e:
                    st.error(f"Login error: {str(e)}")

def register_form_integrated():
    """
    Render registration form.
    """
    st.subheader("Register")

    # Registration form
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")

        if submit:
            # Validate form
            if not username or not email or not password:
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                with st.spinner("Registering..."):
                    try:
                        # Make API request to register
                        response = api_request(
                            "/auth/register",
                            method="POST",
                            data={
                                "username": username,
                                "email": email,
                                "password": password
                            }
                        )

                        if "error" not in response:
                            st.success("Registration successful! You can now login.")
                            # Set show_login to True to switch to login tab
                            st.session_state.show_login = True
                            st.rerun()
                        else:
                            st.error(f"Registration failed: {response['error']}")
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")

def auth_page_integrated():
    """
    Render authentication page with 2FA support.
    """
    st.title("📚 Local File Research")
    st.markdown("*Collaborative research platform for local files*")

    # Add explanation about authentication
    st.info("""
    **Authentication**

    Please login or register to access the application. If you have enabled Two-Factor Authentication (2FA),
    you will be prompted to enter a verification code from your authenticator app after entering your password.

    After logging in, you can access security settings including 2FA setup from the "Security" page in the navigation menu.
    """)

    # Initialize show_login if not in session state
    if "show_login" not in st.session_state:
        st.session_state.show_login = True

    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_form_integrated()

    with tab2:
        register_form_integrated()

def main():
    """
    Main function to render the UI.
    """
    # Initialize session state variables if they don't exist
    if 'report' not in st.session_state:
        st.session_state.report = None

    # Check if user is authenticated
    authenticated = st.session_state.get("authenticated", False)

    # If not authenticated, show auth page
    if not authenticated:
        auth_page_integrated()
        return  # Exit early to avoid rendering the rest of the UI

    # User is authenticated or auth module not available, render the main UI

    # Render sidebar
    sidebar()

    # Render current page
    if st.session_state.current_page == "Research":
        research_page()
    elif st.session_state.current_page == "Documents":
        documents_page()
    elif st.session_state.current_page == "Projects":
        projects_page()
    elif st.session_state.current_page == "Vector Stores":
        vector_stores_page()
    elif st.session_state.current_page == "Settings":
        settings_page()
    elif st.session_state.current_page == "Security":
        security_page()

if __name__ == "__main__":
    # Import and run the main UI function from auth_ui
    #try:
    #    from src.local_file_research.auth_ui import main as auth_main
    #    # Pass the page rendering functions to the auth_main function
    #    auth_main(
    #        page_map={
    #            "Research": research_page,
    #            "Documents": documents_page,
    #            "Projects": projects_page,
    #            "Vector Stores": vector_stores_page,
    #            "Settings": settings_page,
    #            "Security": security_page
    #        },
    #        default_page="Research",
    #        sidebar_func=sidebar # Pass the sidebar function
    #    )
    #except ImportError:
    #    st.error("Failed to import the authentication UI module. Please ensure it's available.")
    #    # Fallback to basic main function if auth_ui is not available
    #    main() # Uncomment this line for fallback behavior
    main()