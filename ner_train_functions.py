import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from spacy import displacy
from spacy.tokens import Doc
from spacy.training import offsets_to_biluo_tags, biluo_to_iob

##### Modified version of the converter from an Iterable of Entities to IOB2 format
##### The results are first converted to the BILOU Schema and then the IOB2 Schema
##### Example: Original ['O', 'B-GPE', 'I-GPE', 'L-GPE', 'O'] => Converted ['O', 'B-GPE', 'I-GPE', 'I-GPE', 'O']
##### To be used to compute A more Lenient Classification Score for entities on a token level
def single_token_tags(doc, entities):
        bilou_tags = offsets_to_biluo_tags(doc, entities)
        iob_tags = biluo_to_iob(bilou_tags)
        return iob_tags


##### Given a sample on our dataset in JSONL format (Prodigy output), get all the true golden entity spans
##### Both the full spans and individual entities are saved into two lists
def get_entities_from_jsonl(jsonl_sample):
        spans = []
        single_entities = []
        for span in jsonl_sample['spans']:
                start, end, label = span['start'], span['end'], span['label']
                single_entities.append({"start": start, "end": end, "label": label})
                spans.append((start, end, label))
        return spans, single_entities
    

##### Displacy visualization of the true entities and the prediction the model is making
##### NOTE: Adapted to be displayed solely on Jupyter Notebooks rather than on the command line
def visualize_predictions(dataset, nlp_model):        
    print('Visualization: ')
    for sample in dataset:
        _, true_entities = get_entities_from_jsonl(sample)
        sentence = sample['text']
        prediction = nlp_model(sentence)
        print('Predicted Entity Spans: ')
        displacy.render(prediction, style = 'ent', jupyter = True)
        true_example = {"text": prediction.text, "ents": true_entities, "title": None}
        print('True Entity Spans: ')
        displacy.render(true_example, style = 'ent', jupyter = True, manual = True)
    
    
##### In the case that our input sentence is missing POS tags, perform an automatic annotation using nltk
##### Given a Sentence String as input, it is tokenized into words and a POS tag is assigned to each word automatically
##### After annotation, the tokens will be converted to a Spacy Doc Object and then returned so a prediction can be made using model(Doc)
def add_pos_tags_to_sentence(spacy_model, sentence):
    words = word_tokenize(sentence, preserve_line = True)
    pos_tags = [tagged_word[1] for tagged_word in nltk.pos_tag(words, tagset = 'universal')]

    #### Assign the POS tags to the tokens in the Doc object
    doc = Doc(spacy_model.vocab, words = words)
    for i, token in enumerate(doc):
        try:
            token.pos_ = pos_tags[i]
        except ValueError:
            token.pos_ = 'PUNCT'
        ##### Debug    
        # print('Word: {} - Assigned POS tag: {}'.format(token, token.pos_))
    return doc