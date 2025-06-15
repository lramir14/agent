import gradio as gr
from chroma_database.manager import ChromaManager
import time

def create_interface():
    db = ChromaManager()
    
    with gr.Blocks(title="RA Local system", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📚 RA for budget analysis\n Welcome to this project in which a RA will help you analyze your pdfs.\nThis system uses a fully local QWEN3 model quantized in 4 bits.") 
        
        # Upload Section
        with gr.Tab("📤 Upload Documents to our knowledge database to help analyze budgets."):
            with gr.Row():
                #with gr.Column():I'm pausing this becuase i yet dont have the capability to enable fast csv files. 
                 #   csv_upload = gr.File(label="CSV File", file_types=[".csv"])
                  #  csv_status = gr.Textbox(label="Status", interactive=False)
                   # csv_btn = gr.Button("Upload", variant="primary")
                
                with gr.Column():
                    pdf_upload = gr.File(label="PDF File", file_types=[".pdf"], file_count="single")
                    pdf_status = gr.Textbox(label="Status", interactive=False)
                    pdf_btn = gr.Button("📎 Upload PDF", variant="primary")
        
        # QA Section
        with gr.Tab("### ❓ Ask Questions..."):
            gr.Markdown("### Type a question about your uploaded documents and get smarter answers")
            
            with gr.Row():
                with gr.Column(scale=3):
                    question=gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., what was the biggest budget cut in 2022?"
                    )
                    ask_btn = gr.Button("🔍 Get Answer", variant="primary")
            with gr.Column(scale=2):
                answer = gr.Textbox(label="RA's Answer", lines=5, interactive=False, show_copy_button=True)
            
            
            
            gr.Markdown("#### 📑 Sources")
            sources = gr.DataFrame(
                headers=["Relevant Excerpts"],
                datatype=["str"],
                interactive=False,
                wrap=True,
                row_count=(2, "dynamic"),
            )
        with gr.Tab("🌐 Web + Local Assistant"):
            gr.Markdown("### 🌍 Ask with Web Search\nWe’ll combine your documents with live web results.")

            with gr.Row():
                web_query = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What did the IMF say about Argentina’s 2024 budget?"
                )
                web_btn = gr.Button("🌐 Search and Answer", variant="primary")
                
            web_answer = gr.Textbox(
                label="Combined Answer",
                lines=6,
                interactive=False,
                show_copy_button=True)

            web_sources = gr.DataFrame(
                headers=["Relevant Snippets (Web + Local)"],
                datatype=["str"],
                interactive=False,
                wrap=True,
                row_count=(2, "dynamic"),)
        
        # Event Handlers
        #def handle_csv(file):
         #   if not file:
          #      return "No file selected"
            
           # start = time.time()
            #processed, uploaded = db.upload_csv(file.name)
            #elapsed = time.time() - start
            
            #return (
             #   f"Processed: {processed} rows | "
              #  f"Uploaded: {uploaded} new | "
               # f"Time: {elapsed:.1f}s"
            #)
        
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
        
        def handle_web_query(query):
            answer_text, sources_data = db.query_with_web(query)
            return answer_text, sources_data
 
        
        #csv_btn.click(handle_csv, inputs=csv_upload, outputs=csv_status)
        pdf_btn.click(handle_pdf, inputs=pdf_upload, outputs=pdf_status)
        ask_btn.click(handle_question, inputs=question, outputs=[answer, sources])
        web_btn.click(handle_web_query, inputs=web_query, outputs=[web_answer, web_sources])
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_port=7860)