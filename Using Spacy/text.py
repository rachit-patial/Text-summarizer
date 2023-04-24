import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# reading the document
def read_doc():
    f = open('text.txt', 'r')
    data = f.read()
    f.close()
    return data

# tokenizing and assigning a score based on the frequency
def scoring():
    text = read_doc()

    stopwords = list(STOP_WORDS)

    # using the en_core_web_sm model model in spacy module to create a NLP pipeline
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    #tokens = [token.text for token in doc]

    # creating a word frequency dictionary
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1


    max_freq = max(word_freq.values())

    # normalizing the freqnecy
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq


    sent_tokens = [sent for sent in doc.sents]

    # assigning the sentence scores
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]
    return sent_tokens, sent_scores, text

# summarizing the text
def summarize():
    sent_tokens, sent_scores, text = scoring()
    select_len = int(len(sent_tokens) * 0.3)

    summary = nlargest(select_len, sent_scores, key = sent_scores.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    print("Length of text: ", len(text))
    print("Length of summary: ", len(summary))

    print(summary)


summarize()