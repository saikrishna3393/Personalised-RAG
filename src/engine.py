from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from utils import pdf_to_text
import string
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def perform_semantic_chunking(doc_path):
    '''
    Performs semantic chunking of a document by extracting text, optionally cleaning it,
    and segmenting it into semantically meaningful chunks.

    :param doc_path: str
        The file path to the document (e.g., a PDF) to be processed for semantic chunking.

    :return: list of str
        A list of text chunks where each chunk represents a semantically meaningful portion of the document.
    '''
    extracted_text = pdf_to_text(doc_path)
    #todo: Text cleaning function to be added
    #cleaned_text = text_cleaning(extracted_text)
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    chunks = semantic_chunker.create_documents([extracted_text])
    lst = []
    for i in chunks:
        lst.append(i.page_content)
    return lst

def create_embeddings_vectors(chunks, query):
    '''
    Creates embedding vectors for text chunks, indexes them in a FAISS vector store, and performs a
    query-based retrieval and generation using an LLM.

    :param chunks: list of str
        A list of text chunks or documents that need to be indexed for retrieval.

    :param query: str
        The input query or prompt that will be used to retrieve relevant documents and generate a response.

    :return: str
        A generated response from the LLM based on the input query and the retrieved relevant documents.
    '''
    ollama_model_name = "llama3.2:1b"
    embeddings = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
    vector_store = FAISS.from_texts(chunks, embeddings)

    llm = OllamaLLM(model=ollama_model_name)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    response = rag_chain.run(query)
    #source_documents = rag_chain.retriever.get_relevant_documents(query)
    print("Query:", query)
    #print("Source Document: ", source_documents)
    print("Generated Response:", response)
    return response


def text_cleaning(text):
    '''
    Cleans the input text by removing punctuation, replacing newlines with spaces, and performing basic normalization.

    :param text: str
        The input text to be cleaned. This can include raw text with punctuation, newlines, or other formatting inconsistencies.

    :return: str
        The cleaned text with punctuation removed and newlines replaced with spaces. The text is also normalized to ensure a cleaner format.
    '''
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

def invoke_rag(query, document):
    '''
        Invokes a Retrieval-Augmented Generation (RAG) pipeline by performing semantic chunking on a document
        and generating a response based on the input query.

        :param query: str
            The input query or question for which a response is required. This query will be used to retrieve
            relevant chunks and generate an answer.

        :param document: str
            The file path to the document (e.g., a PDF) that needs to be processed. The document is
            semantically chunked before generating a response.

        :return: str
            The generated response from the RAG pipeline based on the query and the semantically
            chunked document.
        '''
    chunks_text = perform_semantic_chunking(document)
    return create_embeddings_vectors(chunks_text, query)


# if __name__ == '__main__':
#     chunks_text = perform_semantic_chunking(r"C:\Users\Saikrishna\Downloads\MalwareDetection.pdf")
