from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_text, extract_pages

##### Provided the location of a pdf file on disk, this function extract the entirety of text from it
##### This extracted text can be returned as a single string or a list containing the text of each page
def extract_text_from_pdf(pdf_file):
    page_counter = 0
    text_as_list = []
    text_as_str = ''

    for page_layout in extract_pages(pdf_file):
        page_counter += 1
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                extracted_text = element.get_text()
                text_as_list.append(extracted_text)
                text_as_str += '' + extracted_text
    print('Text from PDF File ({} pages) extracted successfully.'.format(page_counter)) 

    #### High-Level: Do this in a single line (Same Runtime :(, Less Accurate Extraction)
    # text_as_str = extract_text(pdf_file)
    
    return text_as_str, text_as_list