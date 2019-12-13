import pickle
import random
import numpy as np


class DataSupplier:
    def __init__(self, offset=2, num_negative_samples=15):
        self.offset = offset
        self.num_negative_samples = num_negative_samples
        with open('scratchspace/word2token.pkl', 'rb') as f:
            self.word2token = pickle.load(f)

        with open('scratchspace/token2word.pkl', 'rb') as f:
            self.token2word = pickle.load(f)

        with open('scratchspace/tokenized_reviews.pkl', 'rb') as f:
            self.tokenized_reviews = pickle.load(f)
        self.N_WORDS = len(self.token2word)

    def getExample(self, index):
        creview = self.tokenized_reviews[index]
        if len(creview) < (2 * self.offset) + 1:
            return self.getExample((index + 1) % self.N_WORDS)
        r = random.randint(self.offset, len(creview) - self.offset)
        context = creview[r - self.offset:r + self.offset + 1]
        # TODO Improve random selection of word from context.
        neg_samples = []
        nsc = 0
        while nsc < self.num_negative_samples:
            i = random.randint(0, self.N_WORDS - 1)
            if i not in context:
                neg_samples.append(i)
                nsc += 1
        if len(context) == 0:
            return self.getExample(index)
        del context[self.offset]
        context = context[random.randint(0, len(context)) - 1]
        return creview[r], context, neg_samples

    def getExampleFullContext(self, index):
        creview = self.tokenized_reviews[index]
        if len(creview) < (2 * self.offset) + 1:
            return self.getExampleFullContext((index + 1) % self.N_WORDS)
        r = random.randint(self.offset, len(creview) - self.offset)
        context = creview[r - self.offset:r + self.offset + 1]
        # TODO Improve random selection of word from context.
        neg_samples = []
        nsc = 0
        while nsc < self.num_negative_samples:
            i = random.randint(0, self.N_WORDS - 1)
            if i not in context:
                neg_samples.append(i)
                nsc += 1
        if len(context) == 0:
            return self.getExampleFullContext(index)
        del context[self.offset]
        return creview[r], context, neg_samples

    def getBatch(self, numEx):
        words = []
        contexts = []
        otherss = []
        indices = np.random.randint(0, self.N_WORDS, (numEx,))
        for i in indices:
            w, c, os = self.getExample(i)
            words.append(w)
            contexts.append(c)
            otherss.append(os)
        return words, contexts, otherss

    def getBatchFullContext(self, numEx):
        words = []
        contexts = []
        otherss = []
        indices = np.random.randint(0, self.N_WORDS, (numEx,))
        for i in indices:
            w, c, os = self.getExampleFullContext(i)
            words.append(w)
            contexts.append(c)
            otherss.append(os)
        return words, contexts, otherss

    def word(self, token):
        return self.token2word[token]

# ds = DataSupplier(3)
# word, context, neg_sample = ds.getExample(876)
# print(ds.word(word))
# print()
# [print(ds.word(t)) for t in context]
# print()
# [print(ds.word(t)) for t in neg_sample]