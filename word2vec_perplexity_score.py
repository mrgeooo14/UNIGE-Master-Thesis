import random
import math
import numpy as np

##### Assess a single word from the corpus (To be used as a debug method)
def assess_word(model, w, vectors):
    if (w == ''):
        index = random.randint(0, len(vectors))
        w = vectors.index_to_key[index]
    
    print('Given word: {}'.format(w))
    print('Most Similar other words: {}'.format(vectors.most_similar(w, topn = 10)))
    print('Prediction for next word: {} \n'.format([x[0] for x in model.predict_output_word([w], topn = 10)]))


###### Functions related to the calculation of perplexity for our model #####
###### as a way to evaluate our language model on a testing set #############

###### Finding the probability of the target word given the context #########
###### If the word is not found, a uniform probability of 1/V is given ######
###### with V being the number of the total word count ######################
def find_target_proba(corpus_word_count, predictions, target_word, uniforms):
    target_proba = [word[1] for word in predictions if word[0] == target_word]

    if (target_proba == []):  # assume that it exists at least once
        target_proba = [1 / corpus_word_count]
        uniforms.append(1)

    return target_proba[0]


###### The loop to calculate the sum of log likelihoods for each sentence ###
######## the input should be a list of representing the testing set #########
######## This list should contain other lists for each sentence #############
def calculate_perplexity(word2vec_model, testing_set, test_corpus_count, window_size, search_space):
    sentence_n = 0
    global_perplexity = []
    uniforms_n = []

    print('...Perplexity calculation started...')
    for sentence in testing_set:
        sentence_log_likelihoods = []
        sentence_n += 1
        # print('For Sentence #{}: {}'.format(sentence_n, sentence))
        for i in range(0, len(sentence) - math.floor(window_size / 2)):
            window = sentence[i:i + window_size]
            center_word_index = math.floor(len(window) / 2)
            target_word = window[center_word_index]
            # Intentionally hide the target word (kind of MLM) by setting the center target word to an empty string
            window[center_word_index] = ''
            predictions = word2vec_model.predict_output_word(window, topn = search_space)

            if (predictions != None):
                center_word_proba = find_target_proba(test_corpus_count, predictions, target_word,uniforms_n)
                sentence_log_likelihoods.append(np.log(center_word_proba))
            else:
                continue

            perplexity = pow(2, -(np.sum(sentence_log_likelihoods) / len(sentence)))
            global_perplexity.append(perplexity)

        # Debug Mode
        # print('Log Likelihoods {}'.format(sentence_log_likelihoods))
        # print('Local Perplexity: {}'.format(perplexity))
        if (sentence_n % 1000 == 0):
            print('{}\{} sentences processed'.format(sentence_n, len(testing_set)))
    return global_perplexity, uniforms_n