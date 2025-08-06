import gradio as gr
import fitz  # PyMuPDF for PDF parsing
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load PDF content
def load_pdf_text(path):
    try:
        with fitz.open(path) as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e:
        return f" Error loading PDF: {e}"

pdf_text = load_pdf_text("brewmate_manual.pdf")
documents = [Document(page_content=pdf_text)]

# LangChain setup
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Load local Hugging Face LLM
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat function
def chat_with_manual(message, history):
    try:
        response = qa_chain.invoke(message)
        history = history + [[message, response["result"]]]
        return history
    except Exception as e:
        history = history + [[message, f"Error: {str(e)}"]]
        return history

# Gradio UI
with gr.Blocks(css="""
body {
    background: #1e1e1e;
    color: #f2f2f2;
    font-family: 'Segoe UI', sans-serif;
}
.gr-button {
    border-radius: 20px;
    padding: 10px 16px;
    background: #4b6cb7;
    color: white;
    font-weight: bold;
    border: none;
}
.gr-button:hover {
    background: #182848;
    cursor: pointer;
}
#title {
    font-size: 36px;
    font-weight: bold;
    color: #ffffff;
}
#subtitle {
    font-size: 18px;
    color: #aaaaaa;
    margin-bottom: 20px;
}
#examples button {
    margin: 5px;
}
.chatbot {
    border-radius: 10px;
    background: #2e2e2e;
}
.footer {
    display: none !important;
}
""") as demo:

    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("brewmate_logo.png", show_label=False, container=False, height=160)
        with gr.Column(scale=3):
            gr.Markdown("<div id='title'>☕ BrewMate 3000 Help Chat</div>")
            gr.Markdown("<div id='subtitle'>Your friendly assistant for brewing, cleaning, safety, and supplies ✨</div>")

    chatbot = gr.Chatbot(label="BrewMate Assistant", elem_classes=["chatbot"])

    with gr.Row(elem_id="examples"):
        example_questions = [
            "How many cups can it brew?",
            "How do I clean the machine?",
            "What does the red LED mean?",
            "What is the weekly maintenance?"
        ]
        for question in example_questions:
            gr.Button(question).click(fn=chat_with_manual, inputs=[gr.Textbox(value=question, visible=False), chatbot], outputs=chatbot)

    msg = gr.Textbox(placeholder="Type a message and hit Enter...", show_label=False)
    msg.submit(chat_with_manual, inputs=[msg, chatbot], outputs=chatbot)

    gr.Markdown("<div class='footer'>Made with using LangChain + HuggingFace + Gradio</div>")

demo.launch()
