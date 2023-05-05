import re
import time
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_text, extract_pages
from nltk.tokenize import sent_tokenize
import spacy 



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
    return text_as_str, text_as_list

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
    
    return final_keywords

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


def count_entities(doc, entity_labels):
    for ent in doc.ents:
        if ent.label_ in entity_labels:
            yield (ent.label_, ent.text)
            
            
def sort_entity_counts(output_entities):
    sorted_entities = {}
    for entity_type, entity_counts_dict in output_entities.items():
        sorted_entities[entity_type] = dict(sorted(entity_counts_dict.items(), key = lambda item: item[1], reverse=True))
    return sorted_entities

def locate_main_entities(list_of_docs, entity_labels):
    entity_counts = {label: {} for label in entity_labels}
    ###### Loop through the predictions for each sentence and the entities in them
    ###### Entities stored in a Doc object are accessed with doc.ents
    for doc in list_of_docs:
        for label, entity in count_entities(doc, entity_labels):
        # If the entity text has already been seen during the processing, increment its count, else initialize it
            if entity in entity_counts[label]:
                entity_counts[label][entity] += 1
            else:
                entity_counts[label][entity] = 1
                
    entity_counts = sort_entity_counts(entity_counts)
    candidates = []
    for label, entities in entity_counts.items():
        if entities:
            # Find the highest count in the current label
            max_count = max(entities.values())
            entities_in_label = [ent for ent in iter(entities.items()) if ((max_count // ent[1]) < 3)]
            ##### Append the list comprehension to our candidates list
            candidates += [entity[0] for entity in entities_in_label]
                
            # candidates.append(first_element_tuple[0])
            # visualization_text += '{}, '.format(first_element_tuple[0])
            # print("Highest Count {}: {}".format(label, first_element_tuple[0]))
    return candidates



if __name__ == '__main__':
    print("Automatic Named Entity Recognition and Annotation of Digital Health Documents\n")
    start_time = time.time()
    
    nlp_default = spacy.load('en_core_web_lg', enable = ['tok2vec', 'ner'])
    print('Default NER model loaded')
    nlp_finetuned = spacy.load("./models/model-best-lg")
    print('Fine-tuned NER model loaded')
    nlp_roberta = spacy.load('en_core_web_trf', enable = ['transformer', 'ner'])
    print('Transformer (RoBERTa-base) NER model loaded')
    
    pdf_path_name = 'data/implementome_publications/test_miner/child_obesity_switzerland.pdf'
    print("\nPath to PDF given as: {}".format(pdf_path_name))
    
    print('\nDocument Processing Started...')
    single_pdf = open(pdf_path_name, 'rb')
    doc_as_str, doc_as_list = extract_text_from_pdf(single_pdf)
    doc_as_str = remove_references(doc_as_str)
    doc_keywords = try_finding_keywords(doc_as_str)
    print('Keywords Found: {}'.format(doc_keywords))

    print('\nNamed Entity Recognition Started...')
    
    ##### Tokenization and pre-processing of the string containing the full document text
    sentences = sent_tokenize(doc_as_str)
    pattern = r'(?<![a-zA-Z])-|-(?![a-zA-Z])'
    sentences = [re.sub(pattern, ' ', sentence.replace('\n', ' ')) for sentence in sentences]            
                
    our_labels = ['GPE', 'ORG', 'LAW', 'PERSON', 'PRODUCT']

    docs1 = list(nlp_default.pipe(sentences))
    docs2 = list(nlp_finetuned.pipe(sentences))
    docs3 = list(nlp_roberta.pipe(sentences))
    candidates1 = locate_main_entities(docs1, our_labels)
    candidates2 = locate_main_entities(docs2, our_labels)
    candidates3 = locate_main_entities(docs3, our_labels)

    ###### Loop through the predictions for each sentence and the entities in them
    ###### Entities stored in a Doc object are accessed with doc.ents
    
    end_time = time.time()
    print('Full Process finished (for both models)... Time Elapsed: {:.3f} seconds\n'.format(end_time - start_time))
    print('Annotated Entity Candidates (Default Model): {}'.format(', '.join(candidates1)))
    print('Annotated Entity Candidates (Fine-Tuned Model): {}'.format(', '.join(candidates2)))
    print('Annotated Entity Candidates (RoBERTa-base): {}'.format(', '.join(candidates3)))