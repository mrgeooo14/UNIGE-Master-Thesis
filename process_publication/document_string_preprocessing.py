import re

##### Function to locate the possible 'keywords' segment on the string containing the document's text
##### If found, a list that contains all the keywords the authors have put will be generated 
def try_finding_keywords(document_text_str):
    potential_keywords_index = document_text_str.find('Keywords')
    final_keywords = []

    #### Check if the sub-string "Keywords" is located in the document
    if potential_keywords_index != -1:
        found_keywords = []
        for i in range(potential_keywords_index, len(document_text_str)):
            if document_text_str[i] != '\n':
                found_keywords.append(document_text_str[i])
            else:
                break     
                
        keyword = ''
        for i in range(len(found_keywords)):
            if found_keywords[i].isalpha():
                keyword += found_keywords[i]
            elif (keyword != ''):
                final_keywords.append(keyword)
                keyword = ''    
    
    return final_keywords[1:]

##### Function to locate the sub-string holding the References title and remove all the text after it
##### This will potentially remove all the references in a document from the NER pipeline
def remove_references(doc_str):
    print('Original document character length: {}'.format(len(doc_str)))
    potential_references_index = ''
    
    ##### Locate the index of the references title sub-string
    if (doc_str.find('References') != -1) or (doc_str.find('REFERENCES') != -1):
        potential_references_index = doc_str.find('References') if doc_str.find('References') != -1 else doc_str.find('REFERENCES')
        # print('Potential References Index: {}'.format(potential_references_index))
        doc_str = doc_str[:potential_references_index]
        print('References removed, new document character length: {}'.format(len(doc_str)))
    else:
        print('References string index not found')
    return doc_str

##### Remove links from a given string using RE (Regular Expressions)
def remove_links(review):
  review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
  return review