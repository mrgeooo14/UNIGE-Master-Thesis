import gensim
import pandas as pd
from find_similar_tags import similarities_for_tags, not_found_errors

##### Tag Discovery - Pre-trained Biomedical Model
##### Pre-trained biomedical word embeddings from a set of 4billion tokens gathered from PubMed and clinical notes from the MIMIC-III Clinical Database.
##### This one was added to be compared with results from our own models trained from scratch, in terms of the tag discovery process.

# https://github.com/ncbi-nlp/BioSentVec
# Embeddings using PubMed and the clinical notes from MIMIC-III Clinical Database
def load_embeddings():
    pretrained_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        'data/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
        binary = True,
        # limit = None, # this thing has 4 billion tokens (4E9)
        limit=int(5E6) # faster load if you limit to most frequent terms
    ) 
    return pretrained_vectors


if __name__ == "__main__":
    print('Pre-trained biomedical model loaded, discovery process started...')
    biomedical_pretrained_vectors = load_embeddings()
    
    #### Load the given tags that will serve as input to the most_similar() gensim word2vec function
    all_tags_dataframe = pd.read_excel('data/all_tags_word2vec.xlsx')
    one_word_tags = [x for x in all_tags_dataframe['One Word Tags'].values.tolist() if pd.isna(x) == False]
    bigram_tags_dash = [x for x in all_tags_dataframe['Bigram Tags (Med) [Dash]'].values.tolist() if pd.isna(x) == False]
    
    #### Tag Discovery Process
    bigram_similarities = similarities_for_tags(bigram_tags_dash, biomedical_pretrained_vectors)
    one_word_similarities = similarities_for_tags(one_word_tags, biomedical_pretrained_vectors)

    #### Get number of errors (tag not found in our vocabulary)
    #### This means our training set is not sufficient to contain all of our target tags
    not_found_errors(one_word_similarities, bigram_similarities)