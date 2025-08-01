# Deep Research System for Local & Shared Files

<p align="center">
  <img src="assets/dspy-rag-logo.svg" alt="Deep Research System Logo"/>
</p>

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9%2B-purple)
![DSPy](https://img.shields.io/badge/DSPy-2.4%2B-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.7%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

> **🔬 Revolutionary Deep Research System for Local & Shared File Analysis**
>
> **Transform your documents into intelligent knowledge bases.** A cutting-edge, privacy-first platform that performs deep analysis on your local files and shared documents across teams. Features advanced AI research capabilities, multi-iteration analysis, hybrid search, and comprehensive report generation—all while keeping your data completely private and secure on your own system.

---

## 🌟 Revolutionary Deep Research Capabilities

### 🔬 Deep Research System Architecture
**The system operates on both LOCAL FILES and SHARED FILES across users, providing three distinct research modes:**

#### 🚀 Basic RAG (Retrieval-Augmented Generation)
- **Quick Insights**: Fast semantic search with immediate AI-powered analysis
- **Standard Vector Search**: Efficient similarity matching for rapid document discovery
- **Real-time Results**: Instant responses for straightforward research queries
- **Perfect for**: Quick fact-finding, document summaries, basic Q&A

#### 🔄 Hybrid Search (Advanced RAG)
- **Dual-Mode Retrieval**: Combines vector similarity + BM25 text search with QueryFusionRetriever
- **Enhanced Accuracy**: Leverages both semantic understanding and keyword matching
- **Configurable Weights**: Customizable balance between vector (0.6) and BM25 (0.4) search
- **Smart Fusion**: Intelligent result merging with relevance scoring
- **Perfect for**: Complex queries requiring both semantic and lexical matching

#### 🧠 Deep Research Mode (Multi-Iteration Analysis)
- **Iterative Intelligence**: Multi-round research with automatic query refinement and expansion
- **Context Accumulation**: Builds comprehensive understanding across multiple search iterations
- **Follow-up Generation**: AI automatically generates relevant follow-up questions
- **Dynamic Deduplication**: Intelligent removal of duplicate findings across iterations
- **Comprehensive Coverage**: Ensures thorough exploration of research topics
- **Perfect for**: In-depth analysis, research papers, comprehensive investigations

### 📁 Local & Shared File Processing
- **Local File Analysis**: Process your personal documents with complete privacy
- **Shared File Collaboration**: Team-based document sharing with granular permissions
- **Project-Based Organization**: Organize files into collaborative workspaces
- **Cross-User Access**: Seamless file sharing between team members with role-based controls
- **Unified Search**: Search across both personal and shared documents simultaneously

### 🤖 Advanced AI Research Intelligence (15+ Specialized Agents)

#### 📋 Core Analysis Agents
- **🔍 Summarizer**: Intelligent document summarization with key point extraction
- **❓ Answerer**: Context-aware question answering with precise source attribution
- **📊 Extractor**: Structured data extraction from unstructured content
- **🧠 Chain-of-Thought**: Step-by-step reasoning for complex analytical queries
- **✅ Fact Checker**: Information verification and source validation with confidence scoring

#### 🎯 Specialized Document Agents
- **💻 Code Analyzer**: Programming language detection, complexity analysis, documentation generation
- **📈 Spreadsheet Analyzer**: Data pattern recognition, statistical analysis, trend identification
- **📄 PDF Analyzer**: Layout-aware text extraction, table processing, image analysis
- **🔧 Technical Document Analyzer**: API documentation, technical specification analysis
- **🎓 Research Paper Analyzer**: Academic paper structure analysis, citation extraction, methodology review

#### 🚀 Advanced Research Agents
- **🔬 Interpreter**: Deep meaning analysis and contextual interpretation
- **💡 Proposal Generator**: Actionable recommendations and strategic insights
- **⚙️ Technical Analyzer**: Technical feasibility and implementation analysis
- **🔗 Content Synthesizer**: Multi-source information synthesis and cross-referencing
- **📚 Multi-Document Synthesizer**: Cross-document pattern recognition and comprehensive synthesis

### 📊 Comprehensive Report Generation System

#### 📋 Report Types & Modes
- **📄 Normal Reports**: Standard findings with source attribution and basic analysis
- **🧠 Chain-of-Thought Reports**: Detailed reasoning documentation with step-by-step analysis
- **🚀 Enhanced Reports**: Comprehensive analysis including:
  - **🔍 Interpretations**: Deep meaning analysis and context understanding
  - **💡 Proposals**: Actionable recommendations and strategic insights
  - **⚙️ Technical Views**: Implementation details and technical feasibility analysis

#### 📰 Intelligent Article Generation
- **Auto-Detection**: Automatic article type detection based on content analysis
- **📰 Informative Articles**: Well-structured informational content with proper citations
- **📈 Analytical Reports**: Data-driven analysis with insights, trends, and statistical findings
- **📋 Technical Documentation**: Detailed technical specifications, guides, and implementation docs
- **🔍 Research Summaries**: Academic-style research compilation with methodology and conclusions
- **⚖️ Comparative Analyses**: Side-by-side comparison and evaluation with pros/cons
- **🎯 Strategic Recommendations**: Business and strategic guidance with actionable next steps

### 🏗️ Enterprise-Grade Architecture
- **⚡ High-Performance Backend**: FastAPI with async processing and concurrent request handling
- **🖥️ Interactive Frontend**: Streamlit with real-time collaboration and responsive design
- **🗄️ Multi-Vectorstore Support**: FAISS with HNSW/IVF/Flat indices, configurable embedding models
- **🔧 Microservices Design**: Modular architecture with independent service scaling
- **🔒 Privacy-First Processing**: Complete local processing with optional cloud LLM integration

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ with pip
- 8GB+ RAM recommended for optimal performance
- Ollama (optional, for local embeddings): [Install Ollama](https://ollama.ai/)

### Installation & Setup

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd local_file_research
   pip install -r requirements.txt
   ```

2. **Configure Environment** (Optional)
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

3. **Start the System**
   ```bash
   python -m local_file_research.main_llamaindex both
   ```

### 🌐 Access Points
- **🔍 Main Research Interface**: [http://localhost:8501](http://localhost:8501)
- **🔐 Authentication & Security**: [http://localhost:8502](http://localhost:8502)
- **⚡ API Documentation**: [http://localhost:8006/docs](http://localhost:8006/docs)
- **📊 Health Check**: [http://localhost:8006/health](http://localhost:8006/health)

### Alternative Launch Modes
```bash
# API server only
python -m local_file_research.main_llamaindex api

# UI only (requires API running separately)
python -m local_file_research.main_llamaindex ui

# Run tests
python -m local_file_research.main_llamaindex test

# Database migration
python -m local_file_research.main_llamaindex migrate
```

---

## 🏗️ System Architecture

### Directory Structure
```
local_file_research/
├── src/local_file_research/     # 🧠 Core application code
│   ├── api_*.py                 # 🌐 REST API endpoints
│   ├── ui_*.py                  # 🖥️ Streamlit interfaces
│   ├── dspy_*.py                # 🤖 AI agents and configuration
│   ├── *_manager.py             # 📁 Data management layers
│   └── research_system.py       # 🔬 Core research engine
├── storage/                     # 📄 Document storage
│   ├── documents/               # 📁 User uploaded files
│   ├── projects/                # 👥 Project workspaces
│   └── registry/                # 📋 Metadata and indices
├── embeddings/                  # 🧮 Vector embeddings cache
├── sessions/                    # 💾 Research session data
├── analytics/                   # 📊 Usage and performance data
└── project_indices/             # 🗂️ Vector store indices
```

### Service Architecture
- **🌐 API Layer**: FastAPI with 6 specialized endpoint modules
- **🖥️ UI Layer**: Streamlit with dedicated auth interface
- **🧠 AI Layer**: 15+ DSPy agents for specialized analysis
- **🗄️ Data Layer**: Multi-vectorstore with FAISS backend
- **🔐 Security Layer**: JWT + 2FA authentication system
- **📊 Analytics Layer**: Real-time monitoring and reporting

See [`src/architechture.md`](src/architechture.md) for comprehensive architecture details.

---

## 🎯 Key Functionalities

### 📄 Document Processing & Analysis
- **Multi-Format Support**: PDF, DOCX, PPTX, CSV, JSON, HTML, TXT, code files
- **Intelligent Text Extraction**: Layout-aware processing with metadata preservation
- **Batch Processing**: Efficient handling of large document collections
- **Content Analysis**: Automatic language detection, structure analysis, key information extraction

### 🔍 Advanced Search & Retrieval
- **Semantic Search**: Vector similarity search with configurable embedding models
- **Hybrid Retrieval**: Combined vector + BM25 search with QueryFusionRetriever
- **Multi-Iteration Research**: Iterative query refinement with context accumulation
- **Smart Filtering**: Metadata-based filtering and result ranking
- **Context Preservation**: Session-based search history and context maintenance

### 🤖 AI-Powered Analysis (15+ Specialized Agents)

#### Core Analysis Agents
- **📝 Summarizer**: Intelligent document summarization with key point extraction
- **❓ Answerer**: Context-aware question answering with source attribution
- **🔍 Extractor**: Structured data extraction from unstructured content
- **🧠 Chain of Thought**: Step-by-step reasoning for complex queries
- **✅ Fact Checker**: Information verification and source validation

#### Specialized Document Agents
- **💻 Code Analyzer**: Programming language detection, complexity analysis, documentation generation
- **📊 Spreadsheet Analyzer**: Data pattern recognition, statistical analysis, trend identification
- **📄 PDF Analyzer**: Layout-aware text extraction, table processing, image analysis
- **🔧 Technical Document Analyzer**: API documentation, technical specification analysis
- **🎓 Research Paper Analyzer**: Academic paper structure analysis, citation extraction

#### Advanced Research Agents
- **🔬 Interpreter**: Deep meaning analysis and context interpretation
- **💡 Proposal Generator**: Actionable recommendations and strategic insights
- **⚙️ Technical Analyzer**: Technical feasibility and implementation analysis
- **🔗 Content Synthesizer**: Multi-source information synthesis
- **📚 Multi-Document Synthesizer**: Cross-document pattern recognition and synthesis

### 📊 Comprehensive Reporting System

#### Report Types
- **📋 Normal Reports**: Standard findings with basic analysis and source attribution
- **🧠 Chain-of-Thought Reports**: Detailed reasoning documentation with step-by-step analysis
- **🚀 Enhanced Reports**: Comprehensive analysis including:
  - **Interpretations**: Deep meaning analysis and context understanding
  - **Proposals**: Actionable recommendations and strategic insights
  - **Technical Views**: Implementation details and technical feasibility

#### Article Generation
- **📰 Informative Articles**: Well-structured informational content
- **📈 Analytical Reports**: Data-driven analysis with insights and trends
- **📋 Technical Documentation**: Detailed technical specifications and guides
- **🔍 Research Summaries**: Academic-style research compilation
- **⚖️ Comparative Analyses**: Side-by-side comparison and evaluation
- **🎯 Strategic Recommendations**: Business and strategic guidance documents

### 👥 Multi-User & Team Collaboration

#### Project Management
- **🏢 Team Workspaces**: Shared projects with role-based access control
- **👤 User Roles**: Owner, Member, Viewer with appropriate permissions
- **📁 Document Organization**: Hierarchical project structure with tagging
- **🔄 Version Control**: Document versioning and change tracking
- **📋 Activity Feeds**: Real-time project activity and notifications

#### Collaboration Features
- **💬 Comment System**: Document-level and project-level discussions
- **🔄 Real-Time Updates**: Live collaboration with conflict resolution
- **📤 Sharing Controls**: Granular permissions (read/write/admin)
- **🔔 Notifications**: Activity alerts and collaboration updates
- **📊 Team Analytics**: Collaboration metrics and team productivity insights

### 🗄️ Multi-Vectorstore Architecture

#### Vector Store Configurations
- **🚀 FAISS Backend**: High-performance vector similarity search
  - **HNSW Index**: Hierarchical Navigable Small World for fast approximate search
  - **IVF Index**: Inverted File for memory-efficient large-scale search
  - **Flat Index**: Exact search for smaller datasets
- **📏 Distance Metrics**: Inner Product (cosine similarity), L2 (Euclidean distance)
- **💾 Persistence**: Automatic index saving/loading with optimization

#### Embedding Model Support
- **🏠 Local Models**: Ollama integration (mxbai-embed-large, nomic-embed-text)
- **☁️ Cloud Models**: OpenAI, Sentence-transformers, Hugging Face
- **📐 Configurable Dimensions**: 384, 768, 1024, 1536 dimensions
- **⚡ Performance Optimization**: Intelligent caching and batch processing

### 🔐 Security & Authentication Features

#### Authentication System
- **🔑 JWT Tokens**: Secure token-based authentication with configurable expiration
- **📱 Two-Factor Authentication**: TOTP-based 2FA with QR code generation and backup codes
- **👤 User Management**: Registration, login, password management, profile updates
- **🔒 Session Security**: Secure session handling, automatic cleanup, concurrent session management

#### Access Control & Permissions
- **🛡️ Role-Based Access**: Hierarchical permissions (Admin, User, Viewer)
- **🔐 API Key Protection**: Optional API key authentication for enhanced security
- **🌐 CORS Configuration**: Configurable cross-origin resource sharing policies
- **📋 Audit Logging**: Comprehensive security event tracking and monitoring
- **🔍 Input Validation**: Advanced input sanitization and validation

### 📊 Analytics & Monitoring

#### Real-Time Dashboard
- **📈 System Metrics**: Live performance indicators, resource usage, response times
- **👥 User Activity**: Login patterns, feature utilization, collaboration metrics
- **📄 Document Analytics**: Processing statistics, search patterns, popular content
- **🔍 Search Analytics**: Query patterns, result relevance, user satisfaction metrics
- **⚡ Performance Insights**: Bottleneck identification, optimization recommendations

#### Reporting & Export
- **📊 Usage Reports**: Detailed usage statistics and trend analysis
- **📈 Performance Reports**: System performance and optimization insights
- **👥 Team Reports**: Collaboration effectiveness and team productivity
- **📤 Data Export**: CSV, JSON export capabilities for external analysis
- **📅 Scheduled Reports**: Automated report generation and delivery

---

## 💡 Usage Guide

### 🎯 Getting Started
1. **👤 Create Account**: Register at [http://localhost:8502](http://localhost:8502)
2. **🔐 Setup Security**: Enable 2FA for enhanced security
3. **📁 Create Project**: Start a new research project or join existing ones
4. **📄 Upload Documents**: Add your files for analysis
5. **🔍 Start Researching**: Use the main interface for intelligent document analysis

### 🔍 Deep Research Modes & Analysis Options

#### 🎯 Research Mode Selection
- **🚀 RAG Mode**: Standard retrieval-augmented generation for quick insights and immediate answers
- **🔄 Multi-Iteration Mode**: Deep research with iterative query refinement, context accumulation, and follow-up generation
- **🔍 Hybrid Search Mode**: Advanced retrieval combining vector similarity and BM25 text search

#### 📊 Analysis Focus Options
- **📝 Summarize**: Intelligent document summarization with key point extraction
- **❓ Answer**: Context-aware question answering with source attribution
- **🔍 Extract**: Structured data extraction from unstructured content
- **🧠 Chain-of-Thought**: Step-by-step reasoning for complex queries
- **🔬 Analyze**: Comprehensive document analysis with insights and patterns

#### 📋 Report Generation Modes
- **📄 Normal**: Standard findings with basic analysis and source references
- **🧠 Chain-of-Thought**: Detailed reasoning documentation with transparent analysis steps
- **🚀 Enhanced**: Comprehensive reports with interpretations, proposals, and technical views

### 👥 Advanced Team Collaboration & File Sharing

#### 📁 Project-Based File Organization
- **🏢 Team Workspaces**: Create shared projects with hierarchical organization
- **📂 Document Collections**: Organize files into logical groups within projects
- **🔗 Cross-Project Linking**: Reference documents across multiple projects
- **📋 Project Templates**: Standardized project structures for consistent organization

#### 🤝 Multi-User File Sharing System
- **👤 Individual File Access**: Personal document libraries with private access
- **👥 Shared File Access**: Team-based document sharing with granular permissions
- **🔄 Real-Time Synchronization**: Live updates when team members add or modify documents
- **📊 Unified Search**: Search across both personal and shared documents simultaneously
- **🔍 Cross-User Discovery**: Find relevant documents shared by team members

#### 🛡️ Granular Permission System
- **👑 Owner Permissions**: Full control over projects, documents, and team management
- **✏️ Write Permissions**: Edit documents, add new files, and modify project content
- **👁️ Read Permissions**: View documents and research results without modification rights
- **🔒 Private Access**: Personal documents visible only to the owner
- **🔐 Secure Sharing**: Encrypted file sharing with audit trails

#### 💬 Collaborative Research Features
- **📝 Document Annotations**: Add comments and notes directly to documents
- **🗨️ Project Discussions**: Team-wide conversations about research findings
- **📊 Shared Research Sessions**: Collaborative analysis with real-time result sharing
- **🔄 Research History**: Track research evolution and team contributions
- **📈 Team Analytics**: Monitor collaboration patterns and team productivity

### ⚙️ Configuration Options
- **🔧 Environment Variables**: Customize ports, API keys, feature flags via `.env`
- **🤖 AI Models**: Configure embedding models and LLM providers
- **🗄️ Vector Stores**: Choose index types and optimization settings
- **🔐 Security Settings**: Adjust authentication requirements and session policies

---

## 📚 Documentation & Resources

### 📖 Technical Documentation
- **🏗️ Architecture Guide**: [`src/architechture.md`](src/architechture.md) - Comprehensive system architecture
- **🔧 API Documentation**: [http://localhost:8006/docs](http://localhost:8006/docs) - Interactive API explorer
- **⚙️ Configuration Guide**: Environment variables and system configuration
- **🔐 Security Guide**: Authentication, authorization, and security best practices

### 📋 Project Documentation
- **📝 Changelog**: [`CHANGELOG.md`](CHANGELOG.md) - Version history and updates
- **🛡️ Security Policy**: [`SECURITY.md`](SECURITY.md) - Security guidelines and reporting
- **🤝 Code of Conduct**: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - Community guidelines
- **🔧 Contributing Guide**: [`CONTRIBUTING.md`](CONTRIBUTING.md) - Development and contribution guidelines

---

## 🤝 Contributing & Community

### 🚀 How to Contribute
1. **🍴 Fork the Repository**: Create your own copy for development
2. **🌿 Create Feature Branch**: Work on new features in isolated branches
3. **✅ Add Tests**: Ensure your changes include appropriate test coverage
4. **📝 Update Documentation**: Keep documentation current with your changes
5. **🔄 Submit Pull Request**: Submit your changes for review

### 🐛 Reporting Issues
- **🔍 Search Existing Issues**: Check if your issue has already been reported
- **📝 Detailed Reports**: Include steps to reproduce, expected vs actual behavior
- **🏷️ Use Labels**: Help categorize issues (bug, feature request, documentation)
- **🔐 Security Issues**: Report security vulnerabilities privately

### 💬 Community Support
- **❓ Discussions**: Use GitHub Discussions for questions and ideas
- **📖 Documentation**: Check the comprehensive documentation first
- **🔧 Examples**: Look at example configurations and use cases

---

## 🔧 Comprehensive Function Catalog

### 🔬 Core Research Functions

#### Deep Research Pipeline Functions
- **`search_index()`**: Basic RAG search with vector similarity matching
- **`deep_research()`**: Advanced analysis with DSPy agent integration
- **`MultiIterationResearch.conduct_research()`**: Multi-iteration deep research with context accumulation
- **`QueryFusionRetriever.retrieve()`**: Hybrid vector + BM25 search with intelligent fusion
- **`AdvancedReportGenerator.generate_comprehensive_article()`**: Intelligent article generation with auto-type detection

#### Document Processing Functions
- **`process_document()`**: Multi-format document processing with metadata extraction
- **`build_index()`**: Document indexing with embedding generation and vector store creation
- **`DocumentProcessor.process()`**: Advanced text extraction with layout awareness
- **`get_embeddings()`**: Embedding generation with multiple model support
- **`add_chunks()`**: Vector store population with intelligent chunking

### 👥 Collaboration & Sharing Functions

#### Project Management Functions
- **`create_project()`**: Team workspace creation with role-based access
- **`add_member()`**: Team member management with permission assignment
- **`share_project()`**: Project sharing with granular permission control
- **`get_user_projects()`**: Retrieve projects with access level information
- **`add_document()`**: Document addition to projects with metadata tracking

#### File Sharing Functions
- **`create_share()`**: Individual file sharing with permission specification
- **`get_shares()`**: Retrieve sharing information for users and projects
- **`check_permission()`**: Access control validation for resources
- **`validate_token()`**: Authentication token verification
- **`get_current_user()`**: User context retrieval for request processing

#### Real-Time Collaboration Functions
- **`add_comment()`**: Document annotation and discussion creation
- **`get_comments()`**: Retrieve comments with threading and history
- **`broadcast_update()`**: Real-time change propagation across users
- **`track_activity()`**: User activity monitoring and logging
- **`sync_research_session()`**: Collaborative research session management

### 🤖 AI Agent Functions

#### Core Analysis Agent Functions
- **`summarizer.predict()`**: Intelligent document summarization
- **`answerer.predict()`**: Context-aware question answering
- **`extractor.predict()`**: Structured data extraction
- **`chain_of_thought.predict()`**: Step-by-step reasoning analysis
- **`fact_checker.predict()`**: Information verification and validation

#### Specialized Agent Functions
- **`code_analyzer.predict()`**: Programming language analysis and documentation
- **`spreadsheet_analyzer.predict()`**: Data pattern recognition and statistical analysis
- **`pdf_analyzer.predict()`**: Layout-aware PDF processing and extraction
- **`technical_analyzer.predict()`**: Technical feasibility and implementation analysis
- **`multi_doc_synthesizer.predict()`**: Cross-document synthesis and pattern recognition

### 📊 Analytics & Monitoring Functions

#### System Analytics Functions
- **`track_event()`**: User action and system event logging
- **`record_metric()`**: Performance and usage metric collection
- **`get_dashboard_data()`**: Real-time dashboard data aggregation
- **`generate_usage_report()`**: Comprehensive usage analytics reporting
- **`monitor_performance()`**: System performance tracking and optimization

#### Research Analytics Functions
- **`track_research_session()`**: Research activity monitoring and analysis
- **`analyze_search_patterns()`**: Search behavior analysis and insights
- **`measure_collaboration_effectiveness()`**: Team productivity assessment
- **`generate_research_insights()`**: Research pattern analysis and recommendations

---

## ❓ Frequently Asked Questions

### 🔒 Privacy & Security
**Q: Is my data completely private?**
A: Yes! 100% local-first architecture. All document processing, embeddings, and analysis happen on your machine. No data leaves your system unless you explicitly configure cloud LLM providers.

**Q: Can I use this system completely offline?**
A: Absolutely! With local embedding models (via Ollama), the entire system runs offline. You only need internet if you choose to use cloud-based LLM providers for enhanced analysis.

**Q: How secure is the authentication system?**
A: Very secure! Features include JWT tokens, 2FA support, role-based access control, session management, and comprehensive audit logging.

### 🚀 Performance & Scalability
**Q: How many documents can the system handle?**
A: The system scales with your hardware. FAISS indices can handle millions of documents efficiently. Performance depends on available RAM and storage.

**Q: What are the system requirements?**
A: Minimum: Python 3.9+, 4GB RAM. Recommended: 8GB+ RAM, SSD storage for optimal performance with large document collections.

### 🔧 Customization & Extension
**Q: Can I add custom AI agents?**
A: Yes! The DSPy agent system is fully extensible. You can create custom agents for domain-specific analysis tasks.

**Q: How do I integrate with external systems?**
A: Use the comprehensive REST API for integration. Full OpenAPI documentation available at `/docs` endpoint.

**Q: Can I use different embedding models?**
A: Absolutely! Support for Ollama, Sentence-transformers, OpenAI, and other embedding providers. Configurable dimensions and metrics.

### 👥 Collaboration & Teams
**Q: How many users can collaborate simultaneously?**
A: No hard limits on concurrent users. Performance scales with your server resources and network capacity.

**Q: Can I set up different permission levels?**
A: Yes! Comprehensive role-based access control with Owner, Member, and Viewer roles, plus granular document-level permissions.

---

## 📄 License & Legal

This project is licensed under the **MIT License** - see [`LICENSE.txt`](LICENSE.txt) for details.

### 🙏 Acknowledgments
- **LlamaIndex**: For the excellent vector store and retrieval framework
- **FastAPI**: For the high-performance web framework
- **Streamlit**: For the intuitive web interface framework
- **DSPy**: For the advanced AI agent capabilities
- **FAISS**: For efficient vector similarity search
- **Ollama**: For local LLM and embedding model support

---

**🌟 Star this repository if you find it useful! 🌟**
