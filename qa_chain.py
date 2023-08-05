import os
 
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


load_dotenv()
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
QDRANT_URL =  os.environ.get('QDRANT_URL')


def get_prompt():
    prompt_template = """Use the following pieces of context to answer the question at the end, if the context does not provide enough information then try to use your knowledge. If you still don't know the answer or you are not sure if it is true, just say that you don't know, don't try to make up an answer.
CONTEXT: 
{context}
------------------
QUESTION: {question}
------------------
Your answer:"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return PROMPT

PROMPT = get_prompt()


def load_chain(model_name=None, collection_name=None):
    """Logic for loading the chain you want to use should go here."""
    if model_name is None:
        model_name = "gpt-3.5-turbo"
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    qdrant_client = QdrantClient(
        QDRANT_URL,
        # prefer_grpc=True,
    )
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings,
        metadata_payload_key="metadata",
        content_payload_key="text",
    )

    retriever = qdrant.as_retriever(search_kwargs={"k": target_source_chunks})
            
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, temperature=0.1)    
    ### Custom prompts
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False, chain_type_kwargs = {"prompt": PROMPT})
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    return qa



