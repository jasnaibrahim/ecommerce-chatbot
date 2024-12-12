from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd
from ecommbot.data_converter import data_converter


load_dotenv()

HF_TOKEN=os.getenv("HF_TOKEN")
ASTRA_DB_API_ENDPOINT =os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION__TOKEN=os.getenv("ASTRA_DB_APPLICATION__TOKEN")
ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions)


def ingestdata(status):
    vstore=AstraDBVectorStore(
        embedding=embeddings,
        collection_name="ecommercechatbot",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION__TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

    storage=status

    if storage == None:
       docs=data_converter()
       inserted_ids=vstore.add_documents(docs)
    else:
       return vstore
    
    return  vstore,inserted_ids


if __name__ == '__main__':
    vstore,inserted_ids=ingestdata(None)
    print(f"inserted {len(inserted_ids)} documents") 
    results=vstore.similarity_search("can you tell me low budget sound bass head")
    for res in results:
        print(f"*{res.page_content}[{res.metadata}]")