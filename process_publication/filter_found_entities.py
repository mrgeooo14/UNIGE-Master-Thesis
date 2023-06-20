import itertools
from thefuzz import fuzz
import time
import re

##### For each prediction (saved in a Doc spaCy format), yield the entities of it (doc.ents)
def count_entities(doc, entity_labels):
    for ent in doc.ents:
        if ent.label_ in entity_labels:
            yield (ent.label_, ent.text)

##### After the entity counts have been computed and saved in a dictionary, sort them by the count, descending            
def sort_entity_counts(output_entities):
    sorted_entities = {}
    for entity_type, entity_counts_dict in output_entities.items():
        sorted_entities[entity_type] = dict(sorted(entity_counts_dict.items(), key = lambda item: item[1], reverse=True))
        sorted_entities[entity_type] = {key: value for key, value in sorted_entities[entity_type].items() if value != 1}        
    return sorted_entities

##### Check that a given string is made up of maximum three words
##### NOTE: This is a check made for any found PERSON entities that limits the names to the full name of the person only
##### 2 Words for the full name and another one for a potential title, e.g. "Dr. John Hoover" would return true
def is_two_words(input_string):
    input_string = input_string.strip()
    words = input_string.split()
    
    if len(words) > 1 and len(words) <= 3 and all(re.match(r'^[a-zA-Z.]+$', word) for word in words):
        return True
    return False


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
    # print('wtf', entity_counts)
    
    labeled_candidates = {}
    for label, entities in entity_counts.items():
        #### If any entities are found
        if entities:
            ##### For our current label, find the entity with the highest count
            max_count = max(entities.values())
            
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
                            
            if label == "PERSON":
                fuzzy_matches = []
                for i, candidate in enumerate(entities_in_label):
                    for e, _ in itertools.islice(entities.items(), i + 1, None):
                        if (fuzz.partial_ratio(candidate, e) == 100) and (is_two_words(e) == True):
                            fuzzy_matches.append(e)
                entities_in_label += fuzzy_matches
                
            ##### Append the list comprehension to our candidates list
            labeled_candidates[label] = entities_in_label  
    
    print("{} total entities predicted".format(total_entity_count))
    return labeled_candidates


def filter_candidate_list(entity_label : str, candidate_list):
    ##### Keep count of all words that will need to be removed
    remove_set = set()
    
    ##### Loop through each word except the final one (it will be checked with all previous ones anyway)
    for i, word in enumerate(candidate_list[:-1]):
        for other_word in candidate_list[i + 1:]:
            similarity = fuzz.partial_ratio(word, other_word)
            ##### 65 Threshold seemed best for now
            if (similarity > 75):
                shorter_word = word if len(word) <= len(other_word) else other_word
                longer_word = other_word if len(word) <= len(other_word) else word
                
                ##### NOTE: Special Case | If the partial token similarity between two disease entities is equal to 100, do not remove it
                ##### This is so sub-categories of diseases are still saved, e.g. "Cancer" and "Breast Cancer"
                if entity_label in ("DISEASE", "PERSON") and similarity == 100:
                    ###### The case where we might be dealing with "Cancer" and "cancer"
                    if not (shorter_word.istitle() and longer_word.istitle()):
                        continue
                
                remove_set.add(shorter_word)
                ### DEBUG
                # print("Removing word '{}' due to similarity with '{}' ({})".format(shorter_word, longer_word, similarity))
    
    ##### Cleaning up the candidate list
    ##### Keeping only the entities outside of the removal set found above            
    filtered_list = [word for word in candidate_list if word not in remove_set]
    
    if entity_label == "PERSON":
        filtered_list = [word for word in candidate_list if is_two_words(word)]

    return filtered_list
