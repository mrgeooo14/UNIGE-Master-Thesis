import time
import pandas as pd
import sys
import random
import gensim
from gensim.models import Word2Vec
import re

#### Script used to create a custom and random evaluation set for a gensim model ####
#### Given a trained model using gensim's Word2Vec, the custom set will be created ####
#### with the following format of tuples: ([a list of words], the "odd one out"] ####
#### The odd one out is selected by picking one word randomly that is the least similar, picked from the inverse of the list given by model.wv.most_similar() ####
#### These tuples could be used to test the similarity of your gensim model using the function model.wv.most_similar() ####


def word2vec_train_model(model, set, e):
    model.build_vocab(set, progress_per = 800)
    model.train(set, total_examples = model.corpus_count, epochs = e)
    print('Total Words in Corpus: ', model.corpus_total_words)

    return model.wv

def remove_links(review):
  review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
  return review

def write_tuples_to_file(word_tuples):
    with open(r'evaluation_dictionary.txt', 'w', encoding="utf-8") as fp:
        fp.write('\n'.join("({}, '{}')".format(x[0], x[1]) for x in word_tuples))


def create_word_tuples(save_to_file, vectors, size):
    print('vectors length', len(vectors))
    if(size > len(vectors)):
        raise('Evaluation set cant be larger than the size of the vector themselves! ({} > {})'.format(
            size, len(vectors)))
    evaluation_words_set = []
    for i in range(size):
        index = random.randint(0, len(vectors) - 1)
        w = vectors.index_to_key[index]

        most_similar = vectors.most_similar(w, topn=sys.maxsize)
        least_similar = list(reversed(most_similar[-50:]))

        least_similar_target = random.choice(least_similar)[0]
        words_set = [most_similar[0][0], most_similar[1][0], most_similar[2][0], least_similar_target]
        random.shuffle(words_set)
        evaluation_words_set.append((words_set, least_similar_target))

    print('An evaluation model from the testing vocabulary of format ([similar words, the odd one], the odd one) was created.')

    if(save_to_file == True):
        write_tuples_to_file(evaluation_words_set)
        print('File was saved locally under name: evaluation_dictionary.txt')
    return evaluation_words_set


if __name__ == "__main__":
    
    ### First Part - Creating a corpus out of the evaluation set
    df = pd.read_table("data/train_sets/evaluation_set.txt", header=None, names=['Sentences'], lineterminator='\n')
    full_text = df.dropna(subset=['Sentences'])
    full_text['Sentences'] = full_text['Sentences'].apply(lambda x: remove_links(x))
    processed_full_text = full_text['Sentences'].apply(lambda x: gensim.utils.simple_preprocess(x, min_len = 3, deacc=True))
    
    train_set = [sentence for sentence in processed_full_text if sentence != [] and len(sentence) > 3]

    ### Word2Vec Training
    epochs_n = 25
    print('Word2Vec Training Started...')
    b1 = time.time()
    model = Word2Vec(vector_size=300, window=8, min_count=5, workers=4,
                     sg=0, hs=0, negative=10, compute_loss=True, epochs=epochs_n)
    output_vectors = word2vec_train_model(model, train_set, epochs_n)
    train_time = time.time() - b1
    print('Training took {} seconds'.format(train_time))

    ### Creating the evaluation_dictionary.txt file used to calculate the odd one out word
    print('Creation of the OddOneOut evaluation set started!')
    evaluation_words_set = create_word_tuples(True, output_vectors, 2000)
