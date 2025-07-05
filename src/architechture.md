# Local File Research System – Architecture Overview

## 1. Entry Point and Startup Flow

### Main Entry
- The program starts with:
  ```
  python -m local_file_research.main_llamaindex both
  ```
- This triggers the `main()` function in `main_llamaindex.py`, which parses the command-line argument (`both`) and calls `run_both()`.

---

## 2. High-Level Flow (`run_both()`)

### a. Configuration and Cleanup
- Attempts to import `API_PORT` from `config.py`.
- Cleans up:
  - Embeddings directory: via `database_cleanup.cleanup_embeddings_directory`
  - Storage/content directory: via `document_cleanup.cleanup_storage_files`
  - Storage/projects directory: via `document_cleanup.cleanup_projects_folder`

### b. API Server
- Starts the API server as a subprocess:
  - Module: `api_llamaindex_main.py`
  - Waits for `/health` endpoint to be ready.

### c. UI Server
- Starts the UI server as a subprocess:
  - Module: `ui_llamaindex.py` (Streamlit app)
  - Sets environment variable `API_URL` for UI to connect to API.
- Starts Auth UI server:
  - Module: `auth_app.py` (Streamlit app on port 8502)
- Waits for UI server to be ready, then opens browser.

---

## 3. Nested File Usage Structure

### 3.1. main_llamaindex.py
- **Imports and uses:**
  - `config.py` (API_PORT)
  - `database_cleanup.py` (cleanup_embeddings_directory)
  - `document_cleanup.py` (cleanup_storage_files, cleanup_projects_folder)
  - `api_llamaindex_main.py` (API server entry)
  - `ui_llamaindex.py` (UI server entry)
  - `auth_app.py` (Auth UI)
  - `litellm_patch.py` (patches for LLM)
  - `test_llamaindex.py` (for tests)
  - `migrate_to_llamaindex.py` (for migration)

### 3.2. api_llamaindex_main.py
- **Likely imports:**
  - `api_llamaindex.py` (core API logic)
  - `api_documents.py`, `api_projects.py`, `api_sessions.py`, `api_auth.py`, `api_admin.py` (API endpoints)
  - `config.py` (configuration)
  - `models.py` (data models)
  - `database_cleanup.py`, `document_cleanup.py` (maintenance)
  - `logging_config.py`, `error_handling.py` (logging and error handling)
  - `analytics.py`, `analytics_dashboard.py` (analytics endpoints)
  - `file_indexer.py`, `document_manager.py`, `document_processor.py`, `embedding.py`, `vector_store.py` (core document/embedding logic)
  - `auth.py`, `auth_ui.py` (authentication)
  - `realtime.py`, `collaboration.py`, `comment_manager.py` (collaboration features)

### 3.3. ui_llamaindex.py
- **Likely imports:**
  - `streamlit`
  - `api_llamaindex.py` (to call API endpoints)
  - `config.py` (API URL)
  - `auth_ui.py` (authentication UI)
  - `comment_manager.py`, `collaboration.py` (collaboration features)
  - `document_manager.py`, `document_processor.py` (document features)
  - `analytics_dashboard.py` (analytics UI)

### 3.4. auth_app.py
- **Likely imports:**
  - `streamlit`
  - `auth.py`, `auth_ui.py` (authentication logic/UI)
  - `config.py` (configuration)

### 3.5. config.py
- **Used by:** almost all modules for configuration constants.

### 3.6. database_cleanup.py, document_cleanup.py
- **Used by:** `main_llamaindex.py`, `api_llamaindex_main.py` for maintenance.

### 3.7. litellm_patch.py
- **Used by:** `main_llamaindex.py` to patch LLM-related issues.

---

## 4. Deep Connections (Nested Usage)

- `main_llamaindex.py` → `api_llamaindex_main.py` → API endpoint modules (`api_documents.py`, `api_projects.py`, etc.) → core logic modules (`document_manager.py`, `embedding.py`, etc.)
- `main_llamaindex.py` → `ui_llamaindex.py` → API client modules → core logic modules
- `main_llamaindex.py` → `auth_app.py` → `auth.py`, `auth_ui.py`
- `main_llamaindex.py` → `database_cleanup.py`, `document_cleanup.py`
- `main_llamaindex.py` → `config.py` (directly and indirectly via all other modules)
- All modules may use `models.py` for data models and `config.py` for configuration.

---

## 5. Detailed Dependency Graph and Import List

Below is a detailed dependency graph and a list of direct/indirect imports for the main modules in the project:

### main_llamaindex.py
- **Direct imports:**
  - litellm_patch
  - config
  - database_cleanup
  - document_cleanup
  - api_llamaindex_main
  - ui_llamaindex
  - test_llamaindex
  - migrate_to_llamaindex
- **Indirect imports (via above):**
  - api_llamaindex_main → api_llamaindex, api_documents, api_projects, api_sessions, api_auth, api_admin, models, document_manager, embedding, etc.
  - ui_llamaindex → auth_ui, comment_manager, collaboration, document_manager, document_processor, analytics_dashboard, etc.

### api_llamaindex_main.py
- **Direct imports:**
  - api_llamaindex
  - config
  - database_cleanup
  - document_cleanup
  - migrate_to_llamaindex
  - logging_config
  - error_handling
- **Indirect imports:**
  - api_llamaindex → pipeline_llamaindex, document_manager, embedding, etc.
  - API endpoint modules → models, storage_manager, auth, document_registry, etc.

### ui_llamaindex.py
- **Direct imports:**
  - streamlit
  - api_llamaindex
  - config
  - auth_ui
  - comment_manager
  - collaboration
  - document_manager
  - document_processor
  - analytics_dashboard
- **Indirect imports:**
  - api_llamaindex → pipeline_llamaindex, document_manager, embedding, etc.

### research_system.py
- **Direct imports:**
  - multi_iteration_research
  - pipeline
  - models

### multi_iteration_research.py
- **Direct imports:**
  - pipeline

### dspy_config.py
- **Direct imports:**
  - config
  - litellm_patch
  - dspy_agents

### project_indexer.py
- **Direct imports:**
  - legacy_cleanup

### document_manager.py
- **Direct imports:**
  - project_index_cleanup

### project_manager.py
- **Direct imports:**
  - project_index_cleanup

### analytics.py
- **Direct imports:**
  - Used by system_metrics.py

### Standalone/Manual Test
- test_multi_iteration.py (not imported elsewhere, only for manual/standalone testing)

### Obsolete
- dspy_config_patch.py (not imported anywhere)

#### Visual Dependency Graph (Textual)

```
main_llamaindex.py
│
├── litellm_patch.py
├── config.py
├── database_cleanup.py
├── document_cleanup.py
├── api_llamaindex_main.py
│   ├── api_llamaindex.py
│   │   ├── pipeline_llamaindex.py
│   │   ├── document_manager.py
│   │   └── embedding.py
│   ├── api_documents.py
│   ├── api_projects.py
│   ├── api_sessions.py
│   ├── api_auth.py
│   ├── api_admin.py
│   ├── models.py
│   ├── document_manager.py
│   ├── embedding.py
│   └── ... (other core modules)
├── ui_llamaindex.py
│   ├── auth_ui.py
│   ├── comment_manager.py
│   ├── collaboration.py
│   ├── document_manager.py
│   ├── document_processor.py
│   └── analytics_dashboard.py
├── auth_app.py
│   ├── auth.py
│   └── auth_ui.py
├── research_system.py
│   ├── multi_iteration_research.py
│   │   └── pipeline.py
│   └── models.py
├── dspy_config.py
│   ├── config.py
│   ├── litellm_patch.py
│   └── dspy_agents.py
├── project_indexer.py
│   └── legacy_cleanup.py
├── document_manager.py
│   └── project_index_cleanup.py
├── project_manager.py
│   └── project_index_cleanup.py
├── analytics.py
│   └── system_metrics.py
└── test_multi_iteration.py (standalone)
```

---

## 6. Import/Reference Search Results: Obsolete vs. Used Files

A full import/reference search was performed to confirm which files are actually used in the codebase and which are obsolete. Here are the results:

| File                        | Used/Referenced? | Status         |
|-----------------------------|------------------|---------------|
| multi_iteration_research.py  | Yes              | Not obsolete  |
| dspy_agents.py              | Yes              | Not obsolete  |
| dspy_config.py              | Yes              | Not obsolete  |
| legacy_cleanup.py           | Yes              | Not obsolete  |
| project_index_cleanup.py    | Yes              | Not obsolete  |
| pipeline_llamaindex.py      | Yes              | Not obsolete  |
| pipeline.py                 | Yes              | Not obsolete  |
| migrate_to_llamaindex.py    | Yes              | Not obsolete  |
| test_multi_iteration.py     | No (only self)   | Standalone test/manual only |
| analytics.py                | Yes              | Not obsolete  |
| dspy_config_patch.py        | No               | Obsolete      |

**Conclusion:**
- Only `dspy_config_patch.py` is truly obsolete/unreferenced.
- All other files are used, either directly or indirectly, in the main application, API, UI, or as part of the research pipeline.

---