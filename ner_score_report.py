from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqclassify
from seqeval.scheme import IOB2

#### NOTE: The functions expect a tag dictionary of 'true' and 'predicted' keys 
#### The keys hold entity true values and predictions according to the IOB2 Schema

#### Remove unwanted labels (set them to Others ('O'), so errors still count)
#### The entity label input is a list of all the entity labels that we want to include in the score e.g. ['ORG', 'PERSON']
def set_tags_to_fixed_labels(entity_labels, tags_lists):
    for tags in tags_lists:
        # print('Before: {}'.format(tags))
        for i in range(len(tags)):
            if tags[i][2:] not in entity_labels:
                tags[i] = 'O'
        # print('After: {} \n'.format(tags))
    return tags_lists


##### NOTE: Experimental, removing the prefixes of the IOB tags (B-, I-) alltogether to create a more lenient scoring system based solely on accuracy per LABEL.
##### Example: While much progress has been made with tuberculosis control, the World Health Organization (WHO) estimates that 9 million people developed tuberculosis.
##### True: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
##### Prediction: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'ORG', 'ORG', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

def remove_iob(tags_list):
    for tags in tags_list:
        for i, tag in enumerate(tags):
            if tag != 'O':
                tags[i] = tag[2:]
    return tags_list


#### Compute and print the evaluation scores using seqeval (F1 Score, Precision and Recall per label, Micro and Macro Averages)
def compute_and_print_strict_scores(iob_tags, entity_labels):
    print('Performance with respect to only our fixed labels: ')
    
    ##### STRICT: seqclassify classification report taking as inputs the true tags in addition to the predicted ones
    strict_score_report = seqclassify(iob_tags['true'], iob_tags['predicted'], mode = 'strict', zero_division = 1, scheme = IOB2)
    
    ##### LENIENT: Detect entities based on a token-level disregarding the start, unit, or end positions of entities
    true_tags_no_iob = remove_iob(iob_tags['true'])
    predicted_tags_no_iob = remove_iob(iob_tags['predicted'])

    true_tags_flat = [tag for seq in true_tags_no_iob for tag in seq]
    predicted_tags_flat = [tag for seq in predicted_tags_no_iob for tag in seq]

    lenient_score_report = classification_report(true_tags_flat, predicted_tags_flat, labels = entity_labels)
    
    ###### Print both classification scores
    print('\nStrict Score Report (Entity Span-Level):')
    print('\n'.join(strict_score_report.splitlines()))
    print('\nLenient Score Report (Token-Level):')
    print('\n'.join(lenient_score_report.splitlines()))
    