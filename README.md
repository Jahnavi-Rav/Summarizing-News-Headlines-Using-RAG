# Summarizing-News-Headlines-Using-RAG

**Headline Summarization and Retrieval-Augmented Generation (RAG)**
This repository contains code for a text summarization and retrieval-augmented generation (RAG) system using state-of-the-art natural language processing models and techniques. The project focuses on summarizing news headlines and generating contextually relevant summaries using the Retrieval-Augmented Generation approach.

**Key Features:**
**Data Preprocessing: **Cleans and preprocesses news headlines from multiple sources using NLTK for text processing tasks such as tokenization, stopword removal, and lemmatization.
**TF-IDF Vectorization: **Utilizes scikit-learn's TF-IDF vectorizer to transform preprocessed headlines into numerical representations for efficient retrieval.
**DPR (Dense Passage Retrieval): **Implements Facebook's DPR model to encode and retrieve relevant headlines based on user queries, enhancing the relevance of generated summaries.
BART Model for Summarization: Applies the BART (Bidirectional and Auto-Regressive Transformers) model for generating abstractive summaries from retrieved headlines.
**RAG Integration:** Implements Retrieval-Augmented Generation (RAG) using Hugging Face's Transformers library, combining a retriever and generator to improve summary quality by incorporating retrieved contexts.

**Dependencies:**
`Python 3.10
PyTorch
Hugging Face Transformers
NLTK (Natural Language Toolkit)
Scikit-learn
Pandas`

**Usage:**
**Clone the repository:**

`git clone https://github.com/your-username/headline-summarization-RAG.git
cd headline-summarization-RAG`

**Install dependencies:**

`pip install -r requirements.txt`

Run the main script:

`python main.py`

Follow the prompts to enter queries and view generated summaries.

Note: Ensure all required models (facebook/dpr-ctx_encoder-single-nq-base, facebook/bart-large-cnn, facebook/rag-token-nq) are downloaded and accessible via Hugging Face's model hub.

**Contributing:**

Contributions are welcome! If you find any issues or have suggestions for improvements, please submit a pull request or open an issue.
