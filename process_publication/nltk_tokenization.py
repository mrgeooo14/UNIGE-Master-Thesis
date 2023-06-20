import re
from nltk.tokenize import sent_tokenize

##### Function that takes as input the string containing the full document text and tokenizes it into sentences
##### Tokenization is performed using the NLTK library
##### After tokenization, a bit of pre-processing is made to fix any potential issues from the tokenization and remove very short sentences
def tokenize_str(doc_str):
    sentences = sent_tokenize(doc_str)
    pattern = r'(?<![a-zA-Z])-|-(?![a-zA-Z])'
    sentences = [re.sub(pattern, ' ', sentence.replace('\n', ' ')) for sentence in sentences if len(sentence) >= 50]     
    # sentences = [sentence for sentence in sentences if len(sentence) >= 50]
    return sentences    