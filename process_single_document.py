import re
import time
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_text, extract_pages
from nltk.tokenize import sent_tokenize
from thefuzz import fuzz
import PyPDF2
import itertools
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

    #### High-Level: Do this in a single line (Same Runtime :(, Less Accurate Extraction)
    # text_as_str = extract_text(pdf_file)
    
    return text_as_str, text_as_list


def extract_text_from_pdf_2(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file, strict=False)
    meta_data = pdf_reader.metadata

    doc_as_str = ''

    x = pdf_reader.numPages
    for i in range(x): ## skip the last page as it usually includes references
        page = pdf_reader.getPage(i)
        page_text = page.extract_text()
        doc_as_str += page_text
        
    print('PDF processed ({} pages)'.format(x))
    
    try:
        doc_author = meta_data['/Author']
        print('Author: {}'.format(doc_author))
        doc_keywords = meta_data['/Keywords']
                    
        pattern = r"\b\w+(?: \w+)*\b"
        keywords_list = re.findall(pattern, doc_keywords)
        print('Keywords: {}'.format(keywords_list))

    except KeyError:
        pass
    
    return doc_as_str, keywords_list

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
    total_entity_count = 0
    ###### Loop through the predictions for each sentence and the entities in them
    ###### Entities stored in a Doc object are accessed with doc.ents
    for doc in list_of_docs:
        total_entity_count += len(doc.ents)
        for label, entity in count_entities(doc, entity_labels):
        # If the entity text has already been seen during the processing, increment its count, else initialize it
            if entity in entity_counts[label]:
                entity_counts[label][entity] += 1
            else:
                entity_counts[label][entity] = 1
                
    entity_counts = sort_entity_counts(entity_counts)
    
    labeled_candidates = {}
    for label, entities in entity_counts.items():
        #### If any entities are found
        if entities:
            ##### For our current label, find the entity with the highest count
            max_count = max(entities.values())
            
            # print("WHAT", highest_count_entity)
            entities_in_label = [entity[0] for entity in iter(entities.items()) if ((max_count // entity[1]) < 3)]
            
            ##### NOTE: Special Case for disease to detect and save disease variations no matter the count
            ##### This is so sub-categories of diseases are still saved, e.g. "Pneumonia" and "Bacterial Pneumonia"
            ##### This is so sub-categories of diseases are still saved, e.g. "Cancer" and "Breast Cancer"
            if label == "DISEASE":
                fuzzy_matches = []
                for i, candidate in enumerate(entities_in_label):
                    for e, _ in itertools.islice(entities.items(), i + 1, None):
                        if fuzz.partial_ratio(candidate, e) == 100:
                            fuzzy_matches.append(e)
                entities_in_label += fuzzy_matches
                
            ##### Append the list comprehension to our candidates list
            labeled_candidates[label] = entities_in_label  
                      
    print("{} total entity counts predicted".format(total_entity_count))
    return labeled_candidates


def filter_candidate_list(entity_label : str, candidate_list):
    ##### Keep count of all words that will need to be removed
    remove_set = set()
    
    ##### Loop through each word except the final one (it will be checked with all previous ones anyway)
    for i, word in enumerate(candidate_list[:-1]):
        for other_word in candidate_list[i + 1:]:
            similarity = fuzz.partial_ratio(word, other_word)
            ##### 65 Threshold seemed best for now
            if (similarity > 65):
                shorter_word = word if len(word) <= len(other_word) else other_word
                longer_word = other_word if len(word) <= len(other_word) else word
                
                ##### NOTE: Special Case | If the partial token similarity between two disease entities is equal to 100, do not remove it
                ##### This is so sub-categories of diseases are still saved, e.g. "Cancer" and "Breast Cancer"
                if entity_label in ("DISEASE", "PERSON") and similarity == 100:
                    ###### The case where we might be dealing with "Cancer" and "cancer"
                    if not (shorter_word.istitle() and longer_word.istitle()):
                        continue
                
                remove_set.add(shorter_word)
                print("Removing word '{}' due to similarity with '{}' ({})".format(shorter_word, longer_word, similarity))
    return [word for word in candidate_list if word not in remove_set]



if __name__ == '__main__':
    print("Automatic Named Entity Recognition and Annotation of Digital Health Documents\n")
    start_time = time.time()
    
    # nlp_default = spacy.load('en_core_web_lg', enable = ['tok2vec', 'ner'])
    # print('Default NER model loaded')
    nlp_finetuned = spacy.load("./models/full-model-lg")
    load_time = time.time() - start_time
    print('Fine-tuned NER (DEFAULT + DISEASES) model loaded | Load Time: {:.3f} seconds'.format(load_time))
    # nlp_roberta = spacy.load('en_core_web_trf', enable = ['transformer', 'ner'])
    # print('Transformer (RoBERTa-base) NER model loaded')
    
    pdf_path_name = 'data/implementome_publications/test_miner/child_obesity_switzerland.pdf'
    print("\nPath to PDF given as: {}".format(pdf_path_name))
    
    processing_start_time = time.time()
    print('\n#########')
    print('Document Pre-Processing Started...')
    single_pdf = open(pdf_path_name, 'rb')
    user_input = int(input("Enter 1 to use pdfminer as OCR or 2 to use PyPDF2: "))
    if user_input == 1:
        doc_as_str, doc_as_list = extract_text_from_pdf(single_pdf)
        doc_keywords = try_finding_keywords(doc_as_str)
    else:
        doc_as_str, doc_keywords = extract_text_from_pdf_2(single_pdf)

    print('OCR Process Finished, PDF converted to text... | {:.3f}'.format(time.time() - processing_start_time))
    doc_as_str = remove_references(doc_as_str)
    print('Keywords Found: {}'.format(doc_keywords))
    print('#########')
    print('Document Pre-Processing Runtime: {:.3f} seconds'.format(time.time() - processing_start_time))

    ner_start_time = time.time()
    print('\n#########')
    print('Named Entity Recognition Started...')
    
    ##### Tokenization and pre-processing of the string containing the full document text
    sentences = sent_tokenize(doc_as_str)
    pattern = r'(?<![a-zA-Z])-|-(?![a-zA-Z])'
    sentences = [re.sub(pattern, ' ', sentence.replace('\n', ' ')) for sentence in sentences]            
                
    our_labels = ['GPE', 'ORG', 'LAW', 'PERSON', 'PRODUCT', 'DISEASE']

    # docs1 = list(nlp_default.pipe(sentences))
    # candidates_per_label1 = locate_main_entities(docs1, our_labels)

    docs2 = list(nlp_finetuned.pipe(sentences))
    candidates_per_label2 = locate_main_entities(docs2, our_labels)
    
    # docs3 = list(nlp_roberta.pipe(sentences))
    # candidates_per_label3 = locate_main_entities(docs3, our_labels)

    ###### Loop through the predictions for each sentence and the entities in them
    ###### Entities stored in a Doc object are accessed with doc.ents
    
    ner_total_time = time.time() - ner_start_time
    
    print('NER Process finished (for both models)... NER Runtime: {:.3f} seconds | Time per Sentence: {:.3f} seconds'.
          format(time.time() - ner_start_time, ner_total_time / len(sentences)))
    print('#########\n')
    # time.sleep(3.0)

    print('#### Processing Results ####')
    # print('Annotated Entity Candidates (Default Model): {}'.format(', '.join(candidates1)))
    print('Main Candidate Entities per Label: ')
    all_candidates = []
    for label, entities in candidates_per_label2.items():
        if label != "DISEASE":
            filtered_entities = filter_candidate_list(label, entities)
        else:
            filtered_entities = filter_candidate_list(label, entities)
        print('{}: {}'.format(label, ', '.join(filtered_entities)))
        all_candidates += filtered_entities

    all_candidates = [candidate if not candidate.islower() else candidate.capitalize() for candidate in all_candidates]   
    print('\nAll Entities: {}'.format(', '.join(all_candidates)))    # print('Annotated Entity Candidates (RoBERTa-base): {}'.format(', '.join(candidates3)))
    print('\nTotal Runtime: {:.3f} seconds'.format(time.time() - start_time))