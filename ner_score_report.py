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


#### Compute and print the evaluation scores using seqeval (F1 Score, Precision and Recall per label, Micro and Macro Averages)
def compute_and_print_strict_scores(iob_tags):
    print('Performance with respect to only our fixed labels: ')
    
    ##### seqclassify classification report taking as inputs the true tags in addition to the predicted ones
    score_report = seqclassify(iob_tags['true'], iob_tags['predicted'], zero_division = 1, scheme = IOB2)
    strict_score_report = seqclassify(iob_tags['true'], iob_tags['predicted'], mode = 'strict', zero_division = 1, scheme = IOB2)

    # print('\n'.join(score_report.splitlines()))
    print('\nStrict Score Report (Entity Span-Level):')
    print('\n'.join(strict_score_report.splitlines()))
    