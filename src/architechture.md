# Deep Research System for Local & Shared Files – Comprehensive Architecture Guide

## 🏗️ System Overview

The Deep Research System is a revolutionary, multi-layered platform specifically designed for comprehensive analysis of both LOCAL FILES and SHARED FILES across teams. This sophisticated system transforms documents into intelligent knowledge bases through advanced AI research capabilities, multi-iteration analysis, and hybrid search technologies.

### 🎯 Core Philosophy & Capabilities
- **🔬 Deep Research Focus**: Three-tier research system (Basic RAG → Hybrid Search → Multi-Iteration Deep Analysis)
- **📁 Local & Shared Files**: Seamless processing of personal documents and team-shared files
- **🤝 Multi-User Collaboration**: Advanced file sharing with granular permissions and real-time collaboration
- **🧠 AI-Powered Intelligence**: 15+ specialized DSPy agents for comprehensive document analysis
- **🔒 Privacy-First**: Complete local processing with optional cloud LLM integration
- **📊 Comprehensive Reporting**: Multiple report types with intelligent article generation
- **🗄️ Multi-Vectorstore**: Advanced vector storage with FAISS backend and hybrid retrieval
- **⚡ Enterprise-Grade**: Scalable architecture supporting concurrent users and large document collections

---

## 🚀 System Startup & Entry Points

### Main Entry Point
```bash
python -m local_file_research.main_llamaindex both
```

This command triggers the complete system initialization through `main_llamaindex.py`, supporting multiple modes:
- `api` - API server only
- `ui` - UI server only
- `both` - Full system (API + UI + Auth UI)
- `test` - Run test suite
- `migrate` - Database migration utilities

### Startup Sequence (`run_both()`)

#### 1. System Preparation
- **Configuration Loading**: Imports settings from `config.py` including ports, API keys, and feature flags
- **Directory Structure**: Creates required directories (`project_indices/`, `sessions/`, `storage/`)
- **Cleanup Operations**:
  - Embeddings cache cleanup via `database_cleanup.cleanup_embeddings_directory`
  - Storage content cleanup via `document_cleanup.cleanup_storage_files`
  - Project storage cleanup via `document_cleanup.cleanup_projects_folder`

#### 2. Service Orchestration
- **API Server** (Port 8006): FastAPI backend with comprehensive REST endpoints
- **Main UI** (Port 8501): Streamlit interface for research and document management
- **Auth UI** (Port 8502): Dedicated authentication and security management interface
- **Health Checks**: Automated service readiness verification with retry logic

---

## 🏛️ Multi-Layered Architecture

### Layer 1: Core Infrastructure
The foundation layer provides essential services and utilities:

#### Configuration Management (`config.py`)
- **Environment Variables**: API keys, ports, feature flags
- **Model Settings**: Embedding dimensions, FAISS index types, LLM configurations
- **Security Settings**: CORS policies, authentication requirements
- **Research Modes**: RAG, multi-iteration, chain-of-thought configurations
- **Report Types**: Normal, enhanced, comprehensive article generation

#### System Maintenance
- **Database Cleanup** (`database_cleanup.py`): Analytics DB, embeddings cache management
- **Document Cleanup** (`document_cleanup.py`): Storage optimization, orphaned file removal
- **LLM Patches** (`litellm_patch.py`): Compatibility fixes for various LLM providers

### Layer 2: Data & Storage Management

#### Document Processing Pipeline
- **File Indexer** (`file_indexer.py`): Multi-format document ingestion (PDF, DOCX, CSV, JSON, HTML, etc.)
- **Document Processor** (`document_processor.py`): Advanced text extraction with metadata preservation
- **Document Manager** (`document_manager.py`): CRUD operations, versioning, access control
- **Storage Manager** (`storage_manager.py`): File system abstraction with project isolation

#### Vector Store Infrastructure
- **LlamaIndex Integration** (`llamaindex_vector_store.py`):
  - FAISS backend with HNSW, IVF, and Flat index support
  - QueryFusionRetriever for hybrid vector + BM25 search
  - Persistent storage with automatic index optimization
- **Legacy Vector Store** (`vector_store.py`): Fallback implementation
- **Embedding System** (`embedding.py`):
  - Ollama integration for local embeddings
  - Sentence-transformers support
  - Caching and batch processing

### Layer 3: API Services (`api_llamaindex_main.py`)

#### Core Research API (`api_llamaindex.py`)
- **Document Indexing**: Batch processing with progress tracking
- **Semantic Search**: Multi-modal retrieval with relevance scoring
- **Deep Research**: Multi-iteration analysis with context accumulation

#### Specialized API Endpoints
- **Document Management** (`api_documents.py`):
  - Upload/download with metadata extraction
  - Bulk operations and batch processing
  - Content analysis and summarization
- **Project Management** (`api_projects.py`):
  - Multi-user project creation and management
  - Document organization and sharing
  - Access control and permissions
- **Session Management** (`api_sessions.py`):
  - Research session persistence
  - Context preservation across queries
  - Session sharing and collaboration
- **Authentication** (`api_auth.py`):
  - JWT-based authentication
  - Two-factor authentication (2FA) support
  - Role-based access control
- **Administration** (`api_admin.py`):
  - System monitoring and metrics
  - User management and analytics
  - System configuration and maintenance

### Layer 4: User Interface & Experience

#### Main Research Interface (`ui_llamaindex.py`)
- **Research Dashboard**: Multi-mode research with real-time results
- **Document Management**: Upload, organize, and analyze documents
- **Project Workspace**: Collaborative project management
- **Vector Store Management**: Multiple vector store configuration
- **Analytics Dashboard**: Usage metrics and performance insights
- **Settings & Configuration**: System customization and preferences

#### Authentication Interface (`auth_app.py`)
- **User Registration**: Account creation with email verification
- **Login System**: Secure authentication with 2FA support
- **Security Settings**: Password management, 2FA setup
- **Account Management**: Profile updates, security preferences

#### Collaboration Features
- **Comment System** (`comment_manager.py`): Document annotations and discussions
- **Real-time Collaboration** (`collaboration.py`): Multi-user project sharing
- **Team Management**: User roles, permissions, and access control

---

## 🔬 Deep Research Intelligence System

### Three-Tier Research Architecture

#### 🚀 Tier 1: Basic RAG (Retrieval-Augmented Generation)
**Function**: `search_index()` in `pipeline_llamaindex.py`
- **Purpose**: Quick insights and immediate answers from document collections
- **Process**: Direct vector similarity search → AI analysis → Formatted results
- **Use Cases**: Fast fact-finding, document summaries, basic Q&A
- **Performance**: Sub-second response times for straightforward queries
- **Best For**: Initial document exploration, quick reference checks

#### 🔄 Tier 2: Hybrid Search (Advanced RAG)
**Function**: `QueryFusionRetriever` in `llamaindex_vector_store.py`
- **Purpose**: Enhanced accuracy through dual-mode retrieval
- **Process**: Parallel vector + BM25 search → Intelligent fusion → Relevance scoring
- **Configuration**: Customizable weights (default: 0.6 vector, 0.4 BM25)
- **Use Cases**: Complex queries requiring both semantic and lexical matching
- **Performance**: Optimized for accuracy over speed
- **Best For**: Nuanced research questions, technical documentation analysis

#### 🧠 Tier 3: Multi-Iteration Deep Research
**Function**: `MultiIterationResearch` class in `multi_iteration_research.py`
- **Purpose**: Comprehensive, iterative analysis with context building
- **Process**:
  1. Initial query analysis and expansion
  2. Multi-round search with context accumulation
  3. Automatic follow-up question generation
  4. Dynamic deduplication and relevance filtering
  5. Comprehensive synthesis and reporting
- **Features**:
  - **Context Accumulation**: Builds understanding across iterations
  - **Query Expansion**: AI-powered query refinement and broadening
  - **Follow-up Generation**: Automatic relevant question creation
  - **Dynamic K-Value**: Adaptive result count based on accumulated context
  - **Deduplication**: Intelligent removal of redundant findings
- **Use Cases**: In-depth research, academic analysis, comprehensive investigations
- **Best For**: Research papers, strategic analysis, thorough document exploration

### 🤖 Advanced DSPy Agent System (`dspy_config.py`, `dspy_agents.py`)

#### 📋 Core Analysis Agents (5 Agents)
- **🔍 Summarizer**: Multi-level document summarization with key point extraction and hierarchical structuring
- **❓ Answerer**: Context-aware question answering with precise source attribution and confidence scoring
- **📊 Extractor**: Structured data extraction from unstructured content with schema validation
- **🧠 Chain of Thought**: Step-by-step reasoning for complex queries with transparent logic paths
- **✅ Fact Checker**: Information verification and source validation with credibility assessment

#### 🎯 Specialized Document Agents (5 Agents)
- **💻 Code Analyzer**: Programming language detection, complexity analysis, documentation generation, security assessment
- **📈 Spreadsheet Analyzer**: Data pattern recognition, statistical analysis, trend identification, anomaly detection
- **📄 PDF Analyzer**: Layout-aware text extraction, table processing, image analysis, metadata extraction
- **🔧 Technical Document Analyzer**: API documentation analysis, technical specification review, compliance checking
- **🎓 Research Paper Analyzer**: Academic paper structure analysis, citation extraction, methodology review, impact assessment

#### 🚀 Advanced Research Agents (5 Agents)
- **🔬 Interpreter**: Deep meaning analysis, contextual interpretation, semantic relationship mapping
- **💡 Proposal Generator**: Actionable recommendations, strategic insights, implementation roadmaps
- **⚙️ Technical Analyzer**: Technical feasibility assessment, implementation analysis, risk evaluation
- **🔗 Content Synthesizer**: Multi-source information synthesis, cross-referencing, knowledge integration
- **📚 Multi-Document Synthesizer**: Cross-document pattern recognition, comprehensive synthesis, meta-analysis

### 📊 Comprehensive Reporting System (`advanced_reporting.py`)

#### 📋 Report Generation Modes
- **📄 Normal Reports**:
  - Standard findings with source attribution
  - Basic analysis and key insights
  - Structured presentation with citations
- **🧠 Chain-of-Thought Reports**:
  - Detailed reasoning documentation
  - Step-by-step analysis process
  - Transparent logic and decision paths
- **🚀 Enhanced Reports**:
  - **🔍 Interpretations**: Deep meaning analysis and context understanding
  - **💡 Proposals**: Actionable recommendations and strategic insights
  - **⚙️ Technical Views**: Implementation details and feasibility analysis

#### 📰 Intelligent Article Generation
**Function**: `generate_comprehensive_article()` with auto-type detection
- **📰 Informative Articles**: Well-structured content with proper citations and references
- **📈 Analytical Reports**: Data-driven analysis with statistical insights and trend identification
- **📋 Technical Documentation**: Detailed specifications, implementation guides, API documentation
- **🔍 Research Summaries**: Academic-style compilation with methodology and conclusions
- **⚖️ Comparative Analyses**: Side-by-side evaluation with pros/cons and recommendations
- **🎯 Strategic Recommendations**: Business guidance with actionable next steps and implementation plans

---

## 📊 Analytics & Monitoring System

### Analytics Engine (`analytics.py`)
- **Usage Tracking**: User activity, feature utilization, performance metrics
- **Performance Monitoring**: Query response times, indexing performance, system resource usage
- **Event Logging**: User actions, system events, error tracking
- **Metrics Collection**: Custom metrics, KPIs, trend analysis

### Analytics Dashboard (`analytics_dashboard.py`)
- **Real-time Metrics**: Live system performance and usage statistics
- **User Activity**: Login patterns, feature usage, collaboration metrics
- **Document Analytics**: Processing statistics, search patterns, popular content
- **Performance Insights**: Response times, throughput, resource utilization
- **Export Capabilities**: Data export, report generation, trend analysis

---

## 🔐 Security & Authentication

### Authentication System (`auth.py`)
- **User Management**: Registration, login, password management
- **JWT Tokens**: Secure token-based authentication with expiration
- **Two-Factor Authentication**: TOTP-based 2FA with QR code generation
- **Session Management**: Secure session handling and cleanup
- **Role-Based Access**: User roles and permission management

### Security Features
- **API Key Protection**: Optional API key authentication for enhanced security
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Input Validation**: Comprehensive input sanitization and validation
- **Audit Logging**: Security event logging and monitoring

---

## 🤝 Advanced Multi-User Collaboration & File Sharing Architecture

### 📁 Local & Shared File Processing System

#### 🏠 Local File Processing
**Components**: `document_manager.py`, `storage_manager.py`, `document_registry.py`
- **Personal Document Libraries**: Private file collections accessible only to the owner
- **Individual Vector Stores**: Isolated embedding spaces for personal research
- **Private Research Sessions**: User-specific research history and context preservation
- **Local File Indexing**: Personal document indexing with metadata extraction
- **Secure Storage**: Encrypted local storage with user-specific access controls

#### 🌐 Shared File Processing
**Components**: `collaboration.py`, `share_manager.py`, `project_manager.py`
- **Project-Based Sharing**: Organized file sharing through collaborative workspaces
- **Cross-User Access**: Seamless access to shared documents across team members
- **Unified Search**: Search across both personal and shared document collections
- **Shared Vector Stores**: Project-specific embedding spaces for team research
- **Collaborative Indexing**: Team-based document processing and analysis

### 🏢 Project-Based Collaboration System

#### 📂 Project Management (`project_manager.py`)
**Functions**: `create_project()`, `add_member()`, `add_document()`, `get_projects()`
- **Team Workspaces**: Hierarchical project organization with document collections
- **Member Management**: Add/remove team members with role-based permissions
- **Document Organization**: Structured file organization within projects
- **Project Templates**: Standardized project structures for consistency
- **Cross-Project Linking**: Reference documents and research across projects

#### 🔗 Advanced Sharing System (`share_manager.py`, `collaboration.py`)
**Functions**: `create_share()`, `share_project()`, `get_shares()`
- **Granular Permissions**: Fine-grained access control (read/write/admin)
- **Share Management**: Create, modify, and revoke document shares
- **Permission Inheritance**: Hierarchical permission propagation
- **Share Tracking**: Comprehensive audit trails for all sharing activities
- **Temporary Access**: Time-limited sharing with automatic expiration

#### 🛡️ Security & Access Control (`security.py`, `auth.py`)
**Functions**: `check_permission()`, `validate_token()`, `get_current_user()`
- **Role-Based Access Control (RBAC)**: Hierarchical permission system
  - **👑 Owner**: Full control over projects, documents, and team management
  - **✏️ Member**: Edit documents, add files, participate in research
  - **👁️ Viewer**: Read-only access to documents and research results
- **Resource-Level Permissions**: Document and project-specific access controls
- **Share-Based Access**: Permission inheritance through sharing relationships
- **Audit Logging**: Comprehensive tracking of all access and modification events

### 💬 Real-Time Collaboration Features

#### 🗨️ Comment & Discussion System (`comment_manager.py`)
**Functions**: `add_comment()`, `get_comments()`, `update_comment()`
- **Document Annotations**: Contextual comments linked to specific document sections
- **Project Discussions**: Team-wide conversations about research findings
- **Threaded Conversations**: Hierarchical comment structures for organized discussions
- **Mention System**: User notifications and tagging for targeted communication
- **Comment History**: Version tracking and edit history for all comments

#### 🔄 Real-Time Synchronization
**Components**: `realtime.py`, WebSocket integration
- **Live Updates**: Real-time document and research result synchronization
- **Concurrent Access**: Multi-user document editing with conflict resolution
- **Activity Feeds**: Live project activity streams and notifications
- **Presence Indicators**: Show active users and their current activities
- **Change Broadcasting**: Instant propagation of updates across team members

### 📊 Collaborative Research Features

#### 🔬 Shared Research Sessions
**Functions**: Research session sharing across team members
- **Team Research**: Collaborative analysis with shared context and findings
- **Research History**: Track research evolution and team contributions
- **Shared Insights**: Cross-pollination of research findings and discoveries
- **Collective Intelligence**: Leverage team knowledge for enhanced analysis

#### 📈 Team Analytics & Monitoring
**Components**: `analytics.py`, team-specific metrics
- **Collaboration Metrics**: Track team productivity and engagement patterns
- **Usage Analytics**: Monitor feature utilization across team members
- **Research Patterns**: Analyze team research behaviors and preferences
- **Performance Insights**: Identify collaboration bottlenecks and optimization opportunities

---

## 🗄️ Multi-Vectorstore Architecture

### Vector Store Support
- **Primary**: LlamaIndex with FAISS backend
  - **Index Types**: HNSW (Hierarchical Navigable Small World), IVF (Inverted File), Flat
  - **Metrics**: Inner Product (cosine similarity), L2 (Euclidean distance)
  - **Persistence**: Automatic saving and loading with optimization
- **Hybrid Search**: QueryFusionRetriever combining vector and BM25 search
- **Fallback**: Legacy vector store implementation for compatibility

### Embedding Models
- **Local Models**: Ollama integration (mxbai-embed-large, nomic-embed-text)
- **Cloud Models**: Sentence-transformers, OpenAI embeddings
- **Configurable Dimensions**: Support for various embedding dimensions (384, 768, 1024, 1536)
- **Caching**: Intelligent embedding caching for performance optimization

---

## 🔧 Comprehensive Function & API Catalog

### 🔬 Deep Research Pipeline Functions

#### Core Research Functions (`pipeline_llamaindex.py`, `research_system.py`)
```python
# Basic RAG Research
search_index(vector_store, query, top_k=5, context_filter=None, session_id=None, project_id=None)

# Advanced Deep Research
deep_research(vector_store, query, top_k=5, mode="summarize", context_filter=None, session_id=None, project_id=None)

# Multi-Iteration Research
MultiIterationResearch.conduct_research(query, research_mode="multi_iteration", top_k=5, max_iterations=3, max_k=50, relevance_threshold=0.7)

# Hybrid Search with Fusion
QueryFusionRetriever.retrieve(query_bundle, similarity_top_k=10, vector_weight=0.6, bm25_weight=0.4)
```

#### Document Processing Functions (`document_processor.py`, `file_indexer.py`)
```python
# Document Processing
process_document(file_path, max_file_size_mb=50) -> Tuple[str, Dict[str, Any]]
DocumentProcessor.process(file_path) -> Tuple[str, DocumentMetadata]

# Index Building
build_index(root_dir=".", file_patterns=None, project_id=None, session_id=None, max_file_size_mb=50)

# Embedding Generation
get_embeddings(texts, model_name="mxbai-embed-large:latest") -> List[List[float]]
get_ollama_embeddings(texts, model_name) -> List[List[float]]
```

#### Vector Store Functions (`llamaindex_vector_store.py`)
```python
# Vector Store Operations
LlamaIndexVectorStore.add_chunks(chunks: List[Dict]) -> None
LlamaIndexVectorStore.search(query_embedding, top_k=5, filters=None, use_fusion_retriever=True) -> List[Dict]
LlamaIndexVectorStore.save() -> None
LlamaIndexVectorStore.load() -> bool
```

### 👥 Collaboration & File Sharing Functions

#### Project Management Functions (`project_manager.py`, `collaboration.py`)
```python
# Project Operations
create_project(name: str, description: str, owner: str) -> Dict[str, Any]
get_projects(username: str = None) -> List[Dict[str, Any]]
get_user_projects(username: str) -> List[Dict[str, Any]]
add_member(project_id: str, username: str) -> Optional[Dict[str, Any]]
remove_member(project_id: str, username: str) -> Optional[Dict[str, Any]]
add_document(project_id: str, document_id: str, title: str) -> Optional[Dict[str, Any]]
```

#### File Sharing Functions (`share_manager.py`)
```python
# Sharing Operations
create_share(project_id: str, sharer: str, recipient: str, permission: str = "read") -> Dict[str, Any]
get_shares(project_id: Optional[str] = None, username: Optional[str] = None) -> List[Dict[str, Any]]
delete_share(share_id: str, username: str) -> bool
share_project(project_id: str, username: str, recipient: str, permission: str = "read") -> Dict[str, Any]
```

#### Security & Access Control Functions (`security.py`, `auth.py`)
```python
# Authentication
create_user(username: str, password: str, email: str = None) -> Dict[str, Any]
authenticate(username: str, password: str) -> Dict[str, Any]
validate_token(token: str) -> Optional[Dict[str, Any]]
get_current_user(token: str = Depends(oauth2_scheme)) -> str

# Authorization
check_permission(user: Dict, resource_type: str, resource_id: str, action: str) -> bool
require_permission(resource_type: str, action: str) -> Callable

# Two-Factor Authentication
setup_2fa(username: str) -> Dict[str, Any]
enable_2fa(username: str, token: str) -> Dict[str, Any]
verify_2fa_token(username: str, token: str) -> bool
```

### 🤖 DSPy Agent Functions (`dspy_agents.py`, `dspy_config.py`)

#### Core Analysis Agents
```python
# Document Analysis
analyze_document(content: str, document_type: str, query: str = None) -> Dict[str, Any]
synthesize_documents(documents: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]

# Agent Registry Operations
DSPyAgentRegistry.register_agent(name: str, agent: Any) -> None
DSPyAgentRegistry.get_agent(name: str) -> Any
DSPyAgentRegistry.call_agent(agent_name: str, inputs: Dict[str, Any]) -> Any
DSPyAgentRegistry.call_chain(chain_name: str, inputs: Dict[str, Any]) -> Any
```

#### Specialized Agent Functions
```python
# Code Analysis
code_analyzer.predict(code: str, language: str, query: str, document: str, content: str, context: str)

# Spreadsheet Analysis
spreadsheet_analyzer.predict(data: str, query: str, document: str, content: str, context: str)

# PDF Analysis
pdf_analyzer.predict(content: str, query: str, document: str, context: str)

# Technical Document Analysis
tech_doc_analyzer.predict(content: str, query: str, document: str, context: str)

# Research Paper Analysis
research_paper_analyzer.predict(content: str, query: str, document: str, context: str)
```

### 📊 Advanced Reporting Functions (`advanced_reporting.py`)

#### Report Generation Functions
```python
# Report Generation
AdvancedReportGenerator.generate_interpretations(findings: List[Dict], query: str) -> Dict[str, Any]
AdvancedReportGenerator.generate_proposals(findings: List[Dict], query: str) -> Dict[str, Any]
AdvancedReportGenerator.generate_technical_view(findings: List[Dict], query: str) -> Dict[str, Any]
AdvancedReportGenerator.generate_comprehensive_synthesis(findings: List[Dict], query: str) -> Dict[str, Any]

# Article Generation
AdvancedReportGenerator.generate_comprehensive_article(
    findings: List[Dict],
    query: str,
    all_queries: List[str] = None,
    interpretations: Dict = None,
    proposals: Dict = None,
    technical_view: Dict = None
) -> Dict[str, Any]
```

### 📈 Analytics & Monitoring Functions (`analytics.py`)

#### Analytics Functions
```python
# Event Tracking
track_event(event_type: str, event_data: Dict[str, Any], username: str = None) -> None
get_events(event_type: str = None, username: str = None, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]

# Metrics Collection
record_metric(metric_name: str, metric_value: float, dimensions: Dict[str, Any] = None) -> None
get_metrics(metric_name: str = None, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]

# Performance Monitoring
record_performance(operation: str, duration_ms: float, details: Dict[str, Any] = None) -> None
get_performance_stats(operation: str = None, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]

# Dashboard Data
get_dashboard_data(time_range: str = "24h") -> Dict[str, Any]
get_usage_stats(username: str = None, time_range: str = "24h") -> Dict[str, Any]
```

### 🌐 API Endpoint Functions

#### Research API (`api_llamaindex.py`)
```python
# Research Endpoints
POST /research - conduct_research(request: ResearchRequest, current_user: str)
POST /documents/index - index_documents(request: IndexingRequest, current_user: str)
GET /health - health_check()
```

#### Document API (`api_documents.py`)
```python
# Document Management
GET /documents - list_documents(project_id: Optional[str], current_user: str)
GET /documents/{document_id} - get_document(document_id: str, current_user: str)
POST /documents/upload - upload_document(file: UploadFile, project_id: Optional[str], current_user: str)
DELETE /documents/{document_id} - delete_document(document_id: str, current_user: str)
```

#### Project API (`api_projects.py`)
```python
# Project Management
GET /projects - list_projects(current_user: str)
GET /projects/{project_id} - get_project(project_id: str, current_user: str)
POST /projects - create_project(request: ProjectCreateRequest, current_user: str)
DELETE /projects/{project_id} - delete_project(project_id: str, current_user: str)
```

---

## 📈 System Dependencies & Data Flow

### Core Data Flow Patterns

#### Research Query Flow
```
User Query → UI → API → Research System → Vector Store → DSPy Agents → Results
```

#### Document Processing Flow
```
File Upload → Document Processor → Embedding Generation → Vector Store → Index Update
```

#### Multi-User Collaboration Flow
```
User Action → Authentication → Authorization → Project Access → Collaboration Engine → Real-time Updates
```

### Dependency Hierarchy

```
🏗️ main_llamaindex.py (Entry Point)
├── 🔧 System Infrastructure
│   ├── config.py (Configuration Management)
│   ├── litellm_patch.py (LLM Compatibility)
│   ├── database_cleanup.py (DB Maintenance)
│   └── document_cleanup.py (Storage Cleanup)
├── 🌐 API Layer (api_llamaindex_main.py)
│   ├── 🔍 Core Research (api_llamaindex.py)
│   │   ├── pipeline_llamaindex.py (Research Pipeline)
│   │   ├── document_manager.py (Document CRUD)
│   │   └── embedding.py (Embedding Generation)
│   ├── 📄 Document API (api_documents.py)
│   ├── 📁 Project API (api_projects.py)
│   ├── 🔐 Auth API (api_auth.py)
│   ├── ⚙️ Admin API (api_admin.py)
│   └── 📊 Session API (api_sessions.py)
├── 🖥️ User Interface (ui_llamaindex.py)
│   ├── 🔐 Authentication UI (auth_ui.py)
│   ├── 💬 Collaboration (collaboration.py)
│   ├── 📝 Comments (comment_manager.py)
│   ├── 📊 Analytics (analytics_dashboard.py)
│   └── 📄 Document Processing (document_processor.py)
├── 🔐 Authentication App (auth_app.py)
│   ├── auth.py (Auth Logic)
│   └── auth_ui.py (Auth Interface)
├── 🧠 AI Research System (research_system.py)
│   ├── 🔄 Multi-Iteration Research (multi_iteration_research.py)
│   ├── 📊 Advanced Reporting (advanced_reporting.py)
│   └── 🤖 DSPy Configuration (dspy_config.py)
│       └── dspy_agents.py (AI Agents)
├── 🗄️ Vector Storage
│   ├── llamaindex_vector_store.py (Primary Vector Store)
│   └── vector_store.py (Legacy Support)
├── 📊 Analytics & Monitoring
│   ├── analytics.py (Analytics Engine)
│   └── system_metrics.py (Performance Metrics)
└── 🛠️ Utilities & Tools
    ├── file_indexer.py (File Processing)
    ├── storage_manager.py (Storage Abstraction)
    ├── models.py (Data Models)
    └── serialization_utils.py (Data Serialization)
```

---

## 🔧 System Configuration & Customization

### Environment Configuration
- **API Ports**: Configurable ports for API (8006), UI (8501), Auth UI (8502)
- **Model Settings**: Embedding model selection, dimension configuration
- **Feature Flags**: Enable/disable specific features (2FA, analytics, collaboration)
- **Performance Tuning**: Vector store optimization, caching settings

### Extensibility Points
- **Custom DSPy Agents**: Add specialized analysis agents for domain-specific tasks
- **Document Processors**: Support for additional file formats and data sources
- **Vector Store Backends**: Plugin architecture for alternative vector stores
- **Authentication Providers**: Integration with external auth systems (LDAP, OAuth)

---

## 📋 Module Status & Maintenance

### Active Core Modules
All modules in the system are actively used and maintained:

| Module Category | Status | Purpose |
|----------------|--------|---------|
| **Core Infrastructure** | ✅ Active | System foundation and configuration |
| **API Services** | ✅ Active | REST API endpoints and business logic |
| **User Interfaces** | ✅ Active | Web-based user interaction |
| **AI & Research** | ✅ Active | Intelligent analysis and research capabilities |
| **Data Management** | ✅ Active | Document processing and storage |
| **Security & Auth** | ✅ Active | User management and security |
| **Analytics** | ✅ Active | System monitoring and insights |
| **Collaboration** | ✅ Active | Multi-user features and sharing |

### Testing & Quality Assurance
- **Unit Tests**: Comprehensive test coverage for core functionality
- **Integration Tests**: End-to-end testing of API and UI workflows
- **Performance Tests**: Load testing and performance benchmarking
- **Security Tests**: Authentication and authorization validation

---