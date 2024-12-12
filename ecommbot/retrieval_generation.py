from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from ecommbot.ingest import ingestdata
from dotenv import load_dotenv
import os

load_dotenv()# set up retriever, limit sources to one


HF_TOKEN=os.getenv("HF_TOKEN")

def generation(vstore):

    retriever=vstore.as_retriever(search_kwargs={"k":3})

    PRODUCT_BOT_TEMPLATE="""
   Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}
    QUESTION: {question}

    YOUR ANSWER:
    
    """
    prompt=ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    llm=HuggingFaceEndpoint(repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
    )
    chain= (

        {"context":retriever,"question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain




if __name__ =='__main__':
    vstore=ingestdata("done")
    chain=generation(vstore)
    print(chain.invoke("can you tell low budget sound bass headset"))
    print(chain.invoke("can you tell best bluetooth buds"))