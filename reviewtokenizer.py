import pickle
import numpy as np
with open('scratchspace/word2token.pkl', 'rb') as f:
    word2token = pickle.load(f)
with open('scratchspace/token2word.pkl', 'rb') as f:
    token2word = pickle.load(f)

for val in word2token:
    print(val)
print()
print(token2word[108])
print('a' in word2token)
setofwords = set(word2token)
print(len(setofwords))
with open('scratchspace/reviews.pkl', 'rb') as f:
    reviews = pickle.load(f)
tokenized_reviews = []
for review in reviews:
    f = True
    tokenized_review = []
    for word in review.split(' '):
        if word in word2token:
            tokenized_review.append(word2token[word])
        else:
            print("Nope on "+word)
            f = False
            break
    if f:
        tokenized_reviews.append(tokenized_review)
        print("Yeah")
print(len(tokenized_reviews))
print(tokenized_reviews[0])
[print(token2word[t], end=' ') for t in tokenized_reviews[0]]
with open('scratchspace/tokenized_reviews.pkl', 'wb') as f:
    pickle.dump(tokenized_reviews, f, -1)
