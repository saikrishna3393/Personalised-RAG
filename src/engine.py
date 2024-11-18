

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from utils import pdf_to_text
import string
#from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def perform_semantic_chunking(doc_path):
    extracted_text = pdf_to_text(doc_path)
    cleaned_text = text_cleaning(extracted_text)
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.split_text(cleaned_text)


    # semantic_chunks = semantic_chunker.create_documents([cleaned_text])
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # chunks = text_splitter.split_text(cleaned_text)
    return semantic_chunks
    # return chunks


# text = "This is a test sentence to generate embeddings."
# embedding_vector = embeddings.embed_text(text)
# print(f"Embedding Vector: {embedding_vector}")


def text_cleaning(text):
    test_str = text.translate(str.maketrans('', '',
                                        string.punctuation))
    print(test_str)
    cleaned_text = test_str.replace("\n", " ")
    #print(test_str)
    return cleaned_text

# def vectorStore(embeddings):
#     index = faiss.IndexFlatL2(len(embed_model.embed_query("hello world")))
#
#     vector_store = FAISS(
#         embedding_function=OpenAIEmbeddings(),
#         index=index,
#         docstore= InMemoryDocstore(),
#         index_to_docstore_id={}
#     )


if __name__ == '__main__':
    print(perform_semantic_chunking(r"C:\Users\Saikrishna\Downloads\BurgersEquation.pdf"))