##### Custom script files related to the project
from document_string_preprocessing import try_finding_keywords, remove_references
from load_and_extract_pdf import extract_text_from_pdf
from nltk_tokenization import tokenize_str
from filter_found_entities import locate_main_entities, filter_candidate_list
from query_for_mesh_terms import get_mesh_tree
##### General Libraries
import time
import spacy 


if __name__ == '__main__':
    print("Automatic Named Entity Recognition and Annotation of Digital Health Documents\n")
    start_time = time.time()
    
    nlp_finetuned = spacy.load("../models/full-model-lg")
    print('Fine-tuned NER (DEFAULT + DISEASES) model loaded | Load Time: {:.3f} seconds'.format(time.time() - start_time))
    
    pdf_path_name = '../data/implementome_publications/test_miner/child_obesity_switzerland.pdf'
    # pdf_path_name = '../data/implementome_publications/test_miner/hypertension_india.pdf'
    print("\nPath to PDF given as: {}".format(pdf_path_name))
    
    processing_start_time = time.time()
    print('\n#########')
    print('Document Pre-Processing Started...')
    single_pdf = open(pdf_path_name, 'rb')
    doc_as_str, doc_as_list = extract_text_from_pdf(single_pdf)
    doc_keywords = try_finding_keywords(doc_as_str)

    print('OCR Process Finished, PDF converted to text... | {:.3f} seconds'.format(time.time() - processing_start_time))
    doc_as_str = remove_references(doc_as_str)
    print('Keywords Found: {}'.format(doc_keywords))
    print('#########')
    print('Document Pre-Processing Runtime: {:.3f} seconds'.format(time.time() - processing_start_time))

    ner_start_time = time.time()
    print('\n#########')
    print('Named Entity Recognition Started...')
    
    ##### Tokenization and pre-processing of the string containing the full document text
    sentences = tokenize_str(doc_as_str)        
    our_labels = ['GPE', 'ORG', 'LAW', 'PERSON', 'DISEASE']

    ##### Make NER Predictions
    docs = list(nlp_finetuned.pipe(sentences))
    candidates_per_label = locate_main_entities(docs, our_labels)

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
    disease_entities = []
    
    for label, entities in candidates_per_label.items():
        filtered_entities = filter_candidate_list(label, entities)
        all_candidates.extend(filtered_entities)
        if label == "DISEASE":
            disease_entities = filtered_entities
        print('{}: {}'.format(label, ', '.join(filtered_entities)))

    print('')
    start = time.time()
    for disease in disease_entities:
        disease_mesh_tree = get_mesh_tree(disease)
        if disease_mesh_tree:
            print("Corresponding MeSh Tree for '{}': {}".format(disease, disease_mesh_tree))
    print("MeSh Querying Time: {:1.3f}s".format(time.time() - start))


    all_candidates = [candidate if not candidate.islower() else candidate.capitalize() for candidate in all_candidates]   
    print('\nTotal Key Entities: {}'.format(', '.join(all_candidates)))
    if doc_keywords != []:
        print('Original Keywords: {}'.format(', '.join(doc_keywords)))
    
    print('Total Runtime: {:.3f} seconds'.format(time.time() - start_time))