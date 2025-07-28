import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pickle

def build_vector_store(
    file_paths: list, 
    model_name: str = 'mixedbread-ai/mxbai-embed-large-v1',
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    vector_store_dir: str = 'vector_store'
    ):
    """
    Builds a FAISS vector store from the content of text files.

    Args:
        file_paths (list): A list of paths to the text files to process.
        model_name (str): The name of the SentenceTransformer model to use for embeddings.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between consecutive chunks.
        vector_store_dir (str): The directory to save the vector store and chunks.
    """
    print("🚀 Starting RAG pipeline build...")
    
    # 1. Load and combine content from all files
    full_text = ""
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                print(f"  -> Reading content from {path}")
                full_text += f.read() + "\n\n"
        except FileNotFoundError:
            print(f"  ⚠️ Warning: File not found at {path}. Skipping.")
            continue
            
    if not full_text.strip():
        print("❌ No text content found to process. Aborting RAG build.")
        return

    # 2. Split the text into chunks
    print(f"  -> Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(full_text)
    print(f"✅ Text split into {len(chunks)} chunks.")

    # 3. Load the embedding model
    print(f"  -> Loading embedding model: '{model_name}'...")
    embedding_model = SentenceTransformer(model_name)
    print("✅ Embedding model loaded.")

    # 4. Create embeddings for each chunk
    print("  -> Creating embeddings for all chunks (this may take a while)...")
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu().numpy() # Move to numpy for FAISS
    print("✅ Embeddings created successfully.")

    # 5. Build and save the FAISS vector store
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)

    index_path = os.path.join(vector_store_dir, 'faiss_index.bin')
    chunks_path = os.path.join(vector_store_dir, 'chunks.pkl')

    print(f"  -> Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to '{index_path}'")

    # 6. Save the text chunks for later retrieval
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ Text chunks saved to '{chunks_path}'")
    print("🎉 RAG pipeline build complete!")


if __name__ == '__main__':
    # This is a test run to demonstrate the agent's functionality.
    # It assumes you have already run the researcher and transcriber agents.
    
    CONTEXT_FILES = [
        os.path.join('data', 'web_context.txt'),
        os.path.join('data', 'youtube_context.txt')
    ]
    
    build_vector_store(CONTEXT_FILES)
