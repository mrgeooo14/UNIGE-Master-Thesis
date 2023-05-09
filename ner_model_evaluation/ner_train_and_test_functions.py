import random
import time
from spacy.training import Example
from ner_test_utils import get_entities_from_jsonl, single_token_tags, visualize_predictions

##### Main function to evaluate a Spacy pretrained model with regard to NER
##### It takes as input a Spacy model (ner_model) and an evaluation set as loaded from Prodigy (evaluation_set)
##### IF *pos_tagging* is enabled, components related to automatic Part of Speech (POS) tagging will be loaded to the model
##### IF *visualize* is enabled and a visualization set is also provided, model predictions and true golden values will be printed in a Jupyter Notebook
def evaluate_model(ner_model, evaluation_set, visualization_set, visualize : bool, pos_tagging : bool):
        all_examples = []
        all_tags = {"true" : [], "predicted": []}
        
        if pos_tagging:
                ner_model.enable_pipe('tagger')
                ner_model.enable_pipe('parser')
                ner_model.enable_pipe('attribute_ruler')


        start_time = time.time()
        print('Model evaluation on the test set started | Visualization: {} | POS Tagging: {}'.format(visualize, pos_tagging))
        for sample in evaluation_set:
                true_entity_spans, true_entities = get_entities_from_jsonl(sample)
                sentence = sample['text']
                
                ##### Perform model prediction (Sentence input can be either a Spacy Doc object or a Text String)
                prediction = ner_model(sentence)
                
                ##### Debug: Uncomment if you want to print POS Tags (The case that automatic_pos_tagging is set to TRUE)
                # for i, token in enumerate(prediction):
                #         print(token.text, token.pos_)
                
                ##### Converting each true and predicted entity span to the IOB2 tag schema (instead of START and END indexes of entities)
                predicted_ent_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in prediction.ents]  
                predicted_ent_tags = single_token_tags(prediction, predicted_ent_spans)
                true_ent_tags = single_token_tags(prediction, true_entity_spans)
                
                ##### Create Examples
                example = Example.from_dict(ner_model.make_doc(sentence), {"entities": true_entity_spans})
                example.predicted = prediction
                all_examples.append(example)
                
                ##### Uncomment for debug        
                # print('For Sentence: {}'.format(sentence))
                # print('True Entity Span: {}'.format(true_entity_spans))
                # print('Model Prediction: {}'.format(predicted_ent_spans))
                # print('Predicted Tags: {}'.format(predicted_ent_tags))
                # print('True Tags: {}\n'.format(true_ent_tags))

                # all_examples.append(Example.from_dict(prediction, {'entities': true_entity_spans}))
                all_tags['true'].append(true_ent_tags)
                all_tags['predicted'].append(predicted_ent_tags)        

        end_time = time.time()
        print('Testing finished...')
        print("Total runtime in seconds: {:.4f}".format(end_time - start_time))
        print('Average Time to process one single sentence: {:.4f} seconds'.format((end_time - start_time) / len(evaluation_set)))

        if visualize:
        ##### Visualization Debug
                visualize_predictions(visualization_set, ner_model)
                        
        return all_tags, all_examples



##### Function to fine-tune a SpaCy Model to better adapt to our task
def fine_tune_model(model, train_set, epochs, batch_size, p_dropout):        
    optimizer = model.resume_training()
    losses_dictionary = {}

    ###### The model is updated by taking a list of Example objects and therefore learn from them
    for i in range(epochs):
        ###### Each iteration, the train data is shuffled to avoid generalizations based on order
        random.shuffle(train_set)
        batches = [train_set[j:j + batch_size] for j in range(0, len(train_set), batch_size)]
        for batch in batches:
            examples_in_batch = []
            for jsonl_sample in batch:
                ###### Example objects require a Doc Object containing the sample's text and the IOB tag representation of it
                ###### Hence, we want to create the Doc Object and IOB tags from the JSONL Prodigy Samples using the functions below
                spans, _ = get_entities_from_jsonl(jsonl_sample)
                doc = model.make_doc(jsonl_sample['text'])
                entity_iob_tags = single_token_tags(doc, spans)
                ###### Convert them to an Example object and add them to a list to be fed into the model after each batch
                examples_in_batch.append(Example.from_dict(doc, {"entities": entity_iob_tags}))
            model.update(examples_in_batch,  sgd = optimizer, losses = losses_dictionary, drop = p_dropout)
        print('Epoch {} | Loss: {}'.format(i + 1, losses_dictionary['ner']))    
    
    return model
    