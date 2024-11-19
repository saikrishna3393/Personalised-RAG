

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from utils import pdf_to_text
import string


embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def perform_semantic_chunking(doc_path):
    extracted_text = pdf_to_text(doc_path)
    #todo: Text cleaning function to be added
    #cleaned_text = text_cleaning(extracted_text)
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    chunks = semantic_chunker.create_documents([extracted_text])
    #todo: convert chunks to embeddings using langchain huggingface embeddings, store the vector embeddings to faiss DB
    return chunks



def text_cleaning(text):
    # test_str = text.translate(str.maketrans('', '',string.punctuation))
    # print(test_str)
    # cleaned_text = test_str.replace("\n", " ")
    # return cleaned_text
    # Step 1: Remove ASCII punctuation
    no_punctuation = text.translate(str.maketrans('', '', string.punctuation))
    # Step 2: Replace newlines with spaces
    replaced_newlines = no_punctuation.replace("\n", " ")
    # Step 3: Handle Unicode punctuation (e.g., “ ” — …)
    #replaced_unicode = re.sub(r'[“”‘’—…]', '', replaced_newlines)
    # Step 4: Normalize extra spaces (including leading/trailing spaces)
    #cleaned_text = ' '.join(replaced_unicode.split())
    return replaced_newlines



if __name__ == '__main__':
    chunks_text = perform_semantic_chunking(r"C:\Users\Saikrishna\Downloads\MalwareDetection.pdf")
    print(type(chunks_text))
    for i in chunks_text:
        print(i)