import gradio as gr
from chroma_database.manager import ChromaManager
import time

def create_interface():
    db = ChromaManager()
    
    with gr.Blocks(title="RA Local system", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📚 RA for budget analysis")
        
        # Upload Section
        with gr.Tab("📤 Upload Documents to our knowledge database to help analyze budgets."):
            with gr.Row():
                with gr.Column():
                    csv_upload = gr.File(label="CSV File", file_types=[".csv"])
                    csv_status = gr.Textbox(label="Status", interactive=False)
                    csv_btn = gr.Button("Upload", variant="primary")
                
                with gr.Column():
                    pdf_upload = gr.File(label="PDF File", file_types=[".pdf"])
                    pdf_status = gr.Textbox(label="Status", interactive=False)
                    pdf_btn = gr.Button("Upload", variant="primary")
        
        # QA Section
        with gr.Tab("❓ Ask Questions about the knowledge database"):
            question = gr.Textbox(label="Question", placeholder="Ask about your documents...")
            answer = gr.Textbox(label="Answer", lines=5, interactive=False)
            sources = gr.DataFrame(
                headers=["Relevant Excerpts"],
                datatype=["str"],
                interactive=False
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
        
        # Event Handlers
        def handle_csv(file):
            if not file:
                return "No file selected"
            
            start = time.time()
            processed, uploaded = db.upload_csv(file.name)
            elapsed = time.time() - start
            
            return (
                f"Processed: {processed} rows | "
                f"Uploaded: {uploaded} new | "
                f"Time: {elapsed:.1f}s"
            )
        
        def handle_pdf(file):
            if not file:
                return "No file selected"
            
            start = time.time()
            total, uploaded = db.upload_pdf(file.name)
            elapsed = time.time() - start
            
            return (
                f"Pages: {total} | "
                f"Uploaded: {uploaded} new | "
                f"Time: {elapsed:.1f}s"
            )
        
        def handle_question(query):
            answer_text, sources_data = db.query(query)
            return answer_text, [[s] for s in sources_data]
        
        csv_btn.click(handle_csv, inputs=csv_upload, outputs=csv_status)
        pdf_btn.click(handle_pdf, inputs=pdf_upload, outputs=pdf_status)
        ask_btn.click(handle_question, inputs=question, outputs=[answer, sources])
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_port=7860)