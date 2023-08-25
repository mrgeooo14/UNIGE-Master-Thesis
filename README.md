# Master Thesis: Custom Named Entity Recognition and Topic Classification for Global Health Publications

Welcome to the GitHub repository for my Master's thesis project on Natural Language Processing (NLP) techniques applied to global health documents at the University of Geneva, Switzerland. 
This repository is organized into different sections, each focusing on a specific aspect of the project. Below is an overview of the project structure and instructions for navigating the repository effectively.

## Project Structure

In an era where information flows ceaselessly and health crises span the globe, harnessing the power of NLP is to optimize the extraction of meaningful insights from vast volumes of global health documents released on a daily basis. The repository is organized in the four main components that made up the thesis:

### 1. Word2Vec Tag Similarity Discovery

Folder: [word2vec Tag Discovery](./word2vec%20Tag%20Discovery)

In the **Word2Vec Tag Similarity Discovery** component, the objective is to identify similar tags based on an initial set of tags using cosine similarity within Word2Vec models. The core concept revolves around comparing tag similarity across models trained with varying corpus sizes—ranging from 100,000 words to 2 million and even 8 billion words. The primary question this component addresses is whether words that are correlated in real-world health contexts also exhibit corresponding mathematical vector correlations in Word2Vec representations.

### 2. Named Entity Recognition (NER) of Global Health Documents

Folder: [Named Entity Recognition](./ner_model_evaluation)

The **Named Entity Recognition (NER)** component involves an model comparison of all available spaCy models on a custom test set formed by the extraction of global health publications in PubMed. This comparison evaluates the differences between CNN and Transformer-based models in order to find the best model that balances speed and accuracy. After careful per-label evaluation, the chosen CNN model is fine-tuned using a custom dataset that was annotated and curated using Prodigy. The resulting model excels at identifying various entity types such as ORG (Organizations), GPE (Geopolitical Entities), LAWs (Public Health Laws), PERSON, and DISEASE—where the DISEASE label was incorporated into NER by further refining the model through additional training on the NCBI dataset.

### 3. Topic Classification

Folder: [Topic Classification](./topic_classification)

Topic classification is another natural language processing task that involves
labelling a given textual content with the corresponding topic that it represents,
essentially text categorization. Therefore, in this sector, the focus shifts to few-shot classification as well as zero-shot single and multi-label classification. A custom test set consisting of 1000 global health sentences is employed, categorized into 50 selected labels. The `facebook/bart-large-mnli` model from HuggingFace's repository serves as the backbone for the zero-shot classification, showcasing its robust performance in this context. Nevertheless, the accuracy of the model comes with a high toll in computational complexity, therefore it stands as its own pipeline, not to be merged with the fast CNN NER model.

### 4. Integrated Pipeline

Folder: [Integrated Pipeline](./process_publication)

The **Integrated Pipeline** encapsulates the end-to-end process of annotating a given PDF containing a global health document. The pipeline spans from initial PDF text extraction, text preprocessing, and tokenization to NER predictions and subsequent sorting of entities by frequency. Entities that surpass a defined filtering threshold are selected for further annotation. This comprehensive pipeline offers a seamless transformation of raw PDF content into a an annotated global health document with named entities. The goal of this is to facilitate research literature indexing, searching, retrieval, and avoid human annotation.
