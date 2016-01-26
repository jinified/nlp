#!/usr/bin/env python
from nltk.util import ngrams
import nltk

def createLM(observation, n):
    return nltk.NgramModel(n, observation, estimator=nltk.MLEProbDist)

def calcLMProb(word, context, lm):
    return lm.prob(word, context)

if __name__ == "__main__":
    sentence = "I don't want to close my eye"
    n = 2
    bigram = ngrams(sentence.split(), n)
    for grams in bigram:
        print(grams)
