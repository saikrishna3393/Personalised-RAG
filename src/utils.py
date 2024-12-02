from pdf2image import convert_from_path
import pytesseract
import re

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def pdf_to_text(pdf_path):
    '''
        Extracts text from a PDF file by converting its pages into images and applying OCR (Optical Character Recognition)
        using Tesseract.

        :param pdf_path: str
            The file path to the PDF document that needs to be processed.

        :return: str
            The extracted text from the PDF, cleaned to remove excessive newlines and leading/trailing whitespace.
    '''
    images = convert_from_path(pdf_path, 500)#, poppler_path=r'C:\Users\Saikrishna\Downloads\poppler2801\poppler-24.08.0\Library\bin')
    full_text = ""
    for image in images:
        page_text = pytesseract.image_to_string(image)
        full_text += page_text + "\n"
    clean_text = re.sub(r'\n+', '\n', full_text).strip()
    return clean_text

if __name__ == '__main__':
    print(pdf_to_text(r"C:\Users\Saikrishna\Downloads\ReinforcedLearning.pdf"))