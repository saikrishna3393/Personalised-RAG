## Project: Personalised RAG 

SRC:
	App.py
	Utils.py
	Engine.py
	
* Utils: PDF to Image functionality, which takes the PDF input and convert into an image
		used pdf2image with poppler path, which takes pdf path and poppler path and convert into an image.
		the converted image is passed to an OCR model (Pytesseract) to read the content from the image (page_text = pytesseract.image_to_string(image))
		Used regular expressions to clean the text basic cleaning such as removing \n and spripping spaces, after this this function will return the txt as whole.
		
* Engine: In this the main purpose is to use langchain framework to embedd the text that we got from the utils and store it in a vector database....
		i have used symantic chunker from langchain to split the documents based on the symantics.
		intially i have called the utils-pfd2text function and passed the pdf to get the text out of it.
		I have used hugging face embedding model to create embeddings from the text, (embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
		Before this i have created a text_cleaning function to handle both ASCII (String Module) and Unicode punctuation(Re), normalize extra spaces, and replace newlines with spaces for cleaner text processing.