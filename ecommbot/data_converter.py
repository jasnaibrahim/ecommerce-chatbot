import pandas as pd
from langchain_core.documents import Document

def data_converter():
    df=pd.read_csv("/home/hp/ecommerce-chatbot/data/flipkart_product_review.csv")
    data=df[['product_title','review']]

    product_list=[]

    for index,row in data.iterrows():
      obj={
        'product_name':row['product_title'],
        'review':row['review']
      }
      product_list.append(obj)

    
    
    docs=[]
    for entry in product_list:
       metadata={"product_name":entry["product_name"]}
       doc=Document(page_content=entry['review'],metadata=metadata)
       docs.append(doc)
    return docs