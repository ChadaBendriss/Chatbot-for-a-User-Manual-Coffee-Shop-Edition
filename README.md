# Chatbot-for-a-User-Manual-Coffee-Shop-Edition-

# AI Chatbot (Coffee Machine Edition)


## How to Run (Windows)

1. **Clone the repo**

   ```bash
   git clone https://github.com/YourUsername/repo.git
   cd repo
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   If blocked, run:

   ```bash
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

3. **Install dependencies**

   ```bash
   pip install langchain langchain-community transformers gradio chromadb PyMuPDF sentence-transformers torch
   ```

4. **Run the chatbot**

   ```bash
   python main.py
   ```

   Then open: `http://127.0.0.1:7860/`

---

## Troubleshooting

- **Script blocked?**
  ```bash
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

- **Install stuck?**
  ```bash
  pip install torch --no-cache-dir --default-timeout=100
  pip install chromadb --no-cache-dir --default-timeout=100
  ```

---


