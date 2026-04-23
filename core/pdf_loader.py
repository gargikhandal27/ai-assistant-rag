import os
from langchain_community.document_loaders import PyPDFLoader     #read pdf and convert into document 
from langchain_text_splitters import RecursiveCharacterTextSplitter             #split in chunks smaller the chunks higher efficiency
from pathlib import Path                #handel file path 
from typing import Set                  #type hitting variable is a set of values 

#read pdfs, skipped already processed ones, return documents
def process_all_pdfs(pdf_directory: str, existing_files: Set[str] = None):
    
    #nothing passed then initialize set to store 
    if existing_files is None:
        existing_files = set()

    all_documents = []          #store all extracted pdf content 
    pdf_dir = Path(pdf_directory)           #string path->path object 

    pdf_files = list(pdf_dir.glob("**/*.pdf"))      #find pdfs recursively 
    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_directory}'")

    skipped = 0
    
    for pdf_file in pdf_files:
        #checking if pdf is already indexed 
        if pdf_file.name in existing_files:
            #already indexed then skip and increase counter 
            print(f"  ⏭  Skipping (already indexed): {pdf_file.name}")
            skipped += 1
            continue

        print(f"\n  Processing: {pdf_file.name}")       #which file is getting processed
        try:
            loader = PyPDFLoader(str(pdf_file))     #reading pdf and loading in documents 
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)     #add pages to main list 
            print(f"  ✓ Loaded {len(documents)} page(s)")

        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {e}")       #handel errors safely 

    print(f"\nSummary: {skipped} skipped | {len(pdf_files) - skipped} new | "           #shows summary skipped files new files, total pages 
          f"{len(all_documents)} total pages loaded")
    return all_documents

#split large documents in chunks 
def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    
    if not documents:       #handel empty inputs 
        print("No new documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(             #creats splitter objects 
        chunk_size=chunk_size,                                     #define overlap,chunk size, length method
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]      #splitting priority paragraph, line, word, character 
    )

    split_docs = text_splitter.split_documents(documents)       #actually split docs
    print(f"Split {len(documents)} page(s) into {len(split_docs)} chunk(s)")        #number of chunks 

    if split_docs:                  #if chunk exist show preview 
        print(f"\nExample chunk:")
        print(f"  Content : {split_docs[0].page_content[:200]}...")
        print(f"  Metadata: {split_docs[0].metadata}")

    return split_docs