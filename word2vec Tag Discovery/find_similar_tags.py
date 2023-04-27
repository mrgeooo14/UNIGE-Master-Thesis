
##### Function to find similar tags from the trained corpus given our existing tags as input. 
##### This is how similar tags are discovered starting from an existing tag for each one of the models.
def similarities_for_tags(tags, model_vectors):
    tags_similarities = {}
    for tag in tags:
        print('For Tag: {}'.format(tag))
        try:
            #### The query is performed through gensim's word2vec function most_similar()
            #### topn is the number of similar tags displayed for each original input tag
            similarity_query = model_vectors.most_similar(tag, topn = 100)
            print('Similarities:')
            for x in similarity_query:
                print('{}'.format(x))
            print('')
            tags_similarities[tag] = [item[0] for item in similarity_query]
        except KeyError or ValueError:
            tags_similarities[tag] = 'Not Found'
            print('Tag not found in vocabulary \n')
    return tags_similarities


def not_found_errors(unigram_dict, bigram_dict):
    #### Get number of errors (tag not found in our vocabulary)
    #### This means our training set is not sufficient to contain all of our target tags
    errors_unigram = len([1 for x in unigram_dict if unigram_dict.get(x) == 'Not Found'])
    errors_bigram = len([1 for x in bigram_dict if bigram_dict.get(x) == 'Not Found'])

    print("{:.2f}% of tags weren't found in the model".format((errors_unigram / len(unigram_dict)) * 100))
    print("{:.2f}% of bigram tags weren't found in the model".format((errors_bigram / len(bigram_dict)) * 100))
