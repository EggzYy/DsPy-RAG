# Local File Research System

<p align="center">
  <img src="assets/dspy-rag-logo.svg" alt="Project Logo" width="120"/>
</p>

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

> **Empower your local research.**
> 
> Local File Research System is a modular, extensible platform for deep analysis, semantic search, and collaborative research on your own files—no cloud required. Built with LlamaIndex, FastAPI, and Streamlit.

---

## 🚀 Features
- **FastAPI-based API** for document and project management
- **Streamlit UI** for interactive research and analytics
- **Semantic search** with embeddings, chunking, and advanced retrieval
- **Multi-iteration research** and chain-of-thought support
- **User authentication** and collaboration tools
- **Analytics and reporting** for insights
- **Easy extensibility** for new research modes and pipelines

---

## 🏁 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the system**
   ```bash
   python -m local_file_research.main_llamaindex both
   ```
   - API: [http://localhost:8006](http://localhost:8006)
   - UI:  [http://localhost:8501](http://localhost:8501)
   - Auth UI: [http://localhost:8502](http://localhost:8502)

---

## 📁 Project Structure

- `src/local_file_research/` – Main source code
- `embeddings/` – Embedding cache files
- `sessions/` – Session data
- `storage/` – Document and project storage
- `analytics/` – Analytics database

See [`architechture.md`](architechture.md) for a detailed architecture and dependency graph.

---

## 💡 Usage Tips
- Use the **UI** for interactive research, analytics, and collaboration.
- Use the **API** for programmatic access, automation, or integration with other tools.
- Configure `.env` for custom ports, API keys, and advanced settings.
- For advanced research, try the **multi-iteration** mode in the UI.

---

## 📝 Documentation
- **Architecture:** [`architechture.md`](architechture.md)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)
- **Security Policy:** [`SECURITY.md`](SECURITY.md)
- **Code of Conduct:** [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 🤝 Contributing
Pull requests and issues are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines. For major changes, open an issue first to discuss your ideas.

---

## ❓ FAQ

**Q: Can I use this system offline?**  
A: Yes! All processing is local. No data leaves your machine.

**Q: How do I add new research modes or pipelines?**  
A: See the architecture and code comments for extension points, or open an issue for guidance.

**Q: Is my data private?**  
A: 100% local-first. No cloud, no tracking.

---

## 📄 License
This project is licensed under the terms of the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
