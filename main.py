import gradio as gr
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Ollama embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Create Chroma vector store (persistent so it survives restarts)
vectorstore = Chroma(
    collection_name="context_store",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Ollama main model
llm = OllamaLLM(model="qwen2.5-coder:0.5b")

# Prompt template for context Q&A
template = """
You are a helpful assistant. Use the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Add user-provided data to vector store
def add_context(text, history):
    if not text.strip():
        return history + [("System", "⚠ No text provided.")]
    doc = Document(page_content=text)
    vectorstore.add_documents([doc])
    vectorstore.persist()
    return history + [("System", "✅ Data added to context.")]

# Chat function
def chat_fn(message, history):
    docs = retriever.get_relevant_documents(message)
    context_text = "\n".join([doc.page_content for doc in docs])
    response = llm.invoke(prompt.format(context=context_text, question=message))
    history.append((message, response))
    return history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🦙 Ollama Chat with Custom Context (Local RAG)")

    with gr.Row():
        context_input = gr.Textbox(label="Add Context Data", placeholder="Paste your data here...")
        add_btn = gr.Button("Add to Context")

    chatbot = gr.Chatbot(label="Chat with Ollama", height=500)
    msg = gr.Textbox(placeholder="Ask a question...", lines=1)
    clear = gr.Button("Clear Chat")

    add_btn.click(fn=add_context, inputs=[context_input, chatbot], outputs=chatbot)
    msg.submit(fn=chat_fn, inputs=[msg, chatbot], outputs=chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
