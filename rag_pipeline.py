# rag_pipeline.py

from document import adv_rag

def run_rag(query):
    result = adv_rag.query(query)
    return result["answer"]