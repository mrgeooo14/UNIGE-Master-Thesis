import PyPDF2
import os

#### Function that takes an input as a folder of .pdf documents and writes them to a text file ##
#### Uses the PyPDF2 library that allows us to read a single page of a pdf document #############
def write_pdfs_to_text_file(pdf_directory: str, text_file_path : str):
    empty_text_file = open(text_file_path, "w", encoding="utf-8")
    
    for file in os.listdir(pdf_directory):
     filename = os.fsdecode(file)
    #  print('what are you', filename)
     if filename.endswith('.pdf'):
        pdf_file = open(pdf_directory + filename, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_file, strict=False)
        x = pdf_reader.numPages
        for i in range(x - 1): ## skip the last page as it usually includes references
            page = pdf_reader.getPage(i)
            page_text = page.extract_text()
            empty_text_file.write(page_text)
        print('{} processed ({} pages)'.format(filename, x))
     else:
        continue
    
#### The two parameters that the function needs are the path of the text file (empty) ###########
#### And the directory that includes the pdf documents that will be read and written ############  
if __name__ == "__main__":
    text_file_path = 'data/train_sets/train_100_books.txt'
    pdfs_directory = 'data/implementome_publications/train_set_100_books/'
    print('Writing PDFs to the provided text file started...')
    write_pdfs_to_text_file(pdfs_directory, text_file_path)
    print('Process finished')