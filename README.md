# IR_Project

## Motivation
The main goal of this chrome extension is to help you get the information you are searching for faster. When you do a google search, the extension will show you the summaries of the articles without the need to open them.

## Installation (Beta Version)
1. Clone the github repo on your local system.
2. Install libraries: `pip instal -r requirements.txt`
3. Download the nltk extensions (stopwords, punkt) (`nltk.download('stopwords')`)
4. Run the Flask server: `python3 app.py` (Optional: You can also specify the summarisation algorithm to use `python3 app.py 1`)
5. Add the extension by going to Manage Extensions > Load Unpacked.
6. Search on google and hover over the title to get the summary.

## Features
* Get summary without opening the article
* Choose between multiple available algorithms based on your usecase or available compute resources
* Preloading for better UX (article summaries are preloaded for faster response)
* Seemless integration in your workflow.

### Algorithms Available
Tip: Enter the corresponding number when running the server to use the algorithm of your choice.

1. TF-IDF
2. LexRank
3. T5 Transformer
4. GPT-2 
5. BERT
6. Gensim
