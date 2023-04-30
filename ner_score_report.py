from seqeval.metrics import classification_report as seqclassify

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
def compute_and_print_scores(entity_labels, prediction_tags):
    print('Performance with only our needed tags: ')
    
    ##### Set any IOB tag that contains a different entity label than our wanted ones to 'O' (Others)
    prediction_tags['true'] = set_tags_to_fixed_labels(entity_labels, prediction_tags['true'])
    prediction_tags['predicted'] = set_tags_to_fixed_labels(entity_labels, prediction_tags['predicted'])
    
    ##### seqclassify classification report taking as inputs the true tags in addition to the predicted ones
    score_report = seqclassify(prediction_tags['true'], prediction_tags['predicted'], zero_division = 1)
    strict_score_report = seqclassify(prediction_tags['true'], prediction_tags['predicted'], mode = 'strict', zero_division = 1)

    print('\nScore Report:')
    print('\n'.join(score_report.splitlines()))

    print('\nStrict Mode:')
    print('\n'.join(strict_score_report.splitlines()))
