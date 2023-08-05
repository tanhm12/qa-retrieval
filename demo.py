import os
import gc
import shutil
import tempfile
from typing import Optional, Tuple
from collections.abc import Iterable

import gradio as gr
from threading import Lock
from typing import List
from tempfile import _TemporaryFileWrapper
from langchain.chains import RetrievalQA

from qa_chain import load_chain
from ingest import main as ingest_files, get_current_files, exist as is_collection_exist, delete_collection, create_collection
from auth import default_auth
from constants import SERVE_PATH, QUEUE_SIZE


from fastapi import FastAPI

app = FastAPI()

model_map = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4"
}
def set_model(_model_name: str, sid: str, pwd: str):
    if default_auth(sid, pwd):
        model_name = model_map[_model_name]
        if is_collection_exist(sid) and len(get_current_files(sid)) > 0:
            return load_chain(model_name, sid)
        else:
            return None
    else:
        return None


def processing_files(files, sid: str,  pwd: str, model_name: str,
                     clear=False):
    print(sid)
    model_name = model_map[model_name]
    if not default_auth(sid, pwd):
        return None, ""
    if clear:
        delete_collection(sid)
        return None, ""
    if files is None or (isinstance(files, Iterable) and len(files) == 0):
        return load_chain(model_name, sid), "\n".join(get_current_files(sid))
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        for file_wrapper in files:
            with open(file_wrapper.name, "rb") as src:
                with open(os.path.join(tmpdirname, os.path.basename(file_wrapper.name)), "wb") as dst:
                    dst.write(src.read())
        ingest_files(sid, tmpdirname)
    
    chain = load_chain(model_name, sid)
    return chain, "\n".join(get_current_files(sid))


def restore_or_create_db(model_name: str, sid: str, pwd: str):
    model_name = model_map[model_name]
    if default_auth(sid, pwd):
        status = ""
        if is_collection_exist(sid):
            status = f"<h3><left>Found existing DB for username {sid}.</left></h3>"
            if len(get_current_files(sid)) > 0:
                chain = load_chain(model_name, sid)
            else:
                chain = None
            return chain, "\n".join(get_current_files(sid)), status
        else:
            status = f"<h3><left>Create new DB for username {sid}.</left></h3>"
            create_collection(sid)
            return None, "", status
    else:
        gr.Error(f"Authentication error!.")   
        status = f"<h3><left><strong>Authentication error!.</strong></left></h3>"
        return None, "", status


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
        
    def prepare_inputs(self, inp: str, history: Optional[Tuple[str, str]]):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history + [[inp, None]]
            return gr.update(value="", interactive=False), history
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        
    def call(
        self, history: Optional[List[List[str]]], chain: Optional[RetrievalQA]
    ):
        global model_name
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            if chain is None:
                raise gr.Error("Please create or restore a DB with some files first!")
            inp, _ = history[-1]
            answer = chain.run(inp)
            history[-1][1] = ""
            for ch in answer:
                history[-1][1] += ch
                yield history
                
        except Exception as e:
            raise e
        finally:
            self.lock.release()

chat = ChatWrapper()

block = gr.Blocks(
    # css=".gradio-container {background-color: lightgray}"
    )

with block:
    with gr.Row():
        gr.Markdown(f"<h1><center>Question Answering Demo</center></h1>")
        
    with gr.Column(scale=0.5):
        gr.Markdown("<h3><left>Enter your account again</left></h3>")    
    with gr.Row():
        with gr.Column(scale=0.5):
            sid = gr.Textbox(
                label="Username"
            )
            pwd = gr.Textbox(
                label="password",
                type="password"
            )
        with gr.Column(scale=0.3, min_width=200):
            restore_sid = gr.Button(value="Create or Restore Database")
            auth_status = gr.Markdown()
        
        # gr.Label("Choose model")
    with gr.Row():
        model = gr.Dropdown(label="Choose model", choices=list(model_map.keys()),  value="gpt-4")

    with gr.Row():
        with gr.Column(scale=0.5):
            file_output = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".docx"])
        with gr.Column(scale=0.1, min_width=200):
            with gr.Row():
                submit_files_btn = gr.Button(value="Process files")
            with gr.Row():
                delete_db_btn = gr.Button(value="Delete Database")
        with gr.Column(scale=0.4):
            current_files = gr.Textbox(
            label="Processed files",
            value="\n".join(get_current_files(sid)),
            interactive=False,
        )
                
         
        
    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="",
            lines=1,
        )
        
    chatbot = gr.Chatbot([])
    chain = gr.State()
    
    txt_msg = message.submit(chat.prepare_inputs, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).then(
        chat.call, [chatbot, chain], chatbot
    )
    # submit.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    txt_msg.then(lambda: gr.update(interactive=True), None, [message], queue=False)
    
    model.change(
        set_model,
        inputs=[model, sid, pwd],
        outputs=[chain],
    )
    restore_sid.click(restore_or_create_db, 
                      inputs=[model, sid, pwd], outputs=[chain, current_files, auth_status])  
    submit_files_btn.click(processing_files, inputs=[file_output, sid, pwd, model], outputs=[chain, current_files])
    delete_db_btn.click(lambda *x: processing_files(*x, clear=True), inputs=[file_output, sid, pwd, model], outputs=[chain, current_files])

# block.auth = default_auth
# block.auth_message = ""

block.queue(QUEUE_SIZE)
block.launch(server_port=10011, auth=default_auth, server_name="0.0.0.0", 
             #root_path=SERVE_PATH
             )


# @app.get("/")
# def read_main():
#     return {"message": "This is your main app"}

# app = gr.mount_gradio_app(app, block, path=SERVE_PATH)

# from _logging import setup_logging
# @app.on_event("startup")
# def startup_event():
#     setup_logging("log/app.log")