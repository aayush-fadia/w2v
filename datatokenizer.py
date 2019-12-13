import pickle
import collections
with open('scratchspace/reviews.pkl', 'rb') as f:
    reviews = pickle.load(f)
print('Number of reviews: {}'.format(len(reviews)))
words = []
for r in reviews:
    [words.append(x) for x in r.split(' ')]
print('Number of words: {}'.format(len(words)))
print(words[0])
counts = collections.Counter(words)
new_list = sorted(words, key=counts.get, reverse=True)
list_from_freq = list(collections.OrderedDict.fromkeys(new_list))
nl = list_from_freq[:10000]
print(nl[998])
word2token = {}
token2word = {}
for i in range(len(nl)):
    word2token[nl[i]] = i
    token2word[i] = nl[i]
with open('scratchspace/word2token.pkl', 'wb') as f:
    pickle.dump(word2token, f, -1)

with open('scratchspace/token2word.pkl', 'wb') as f:
    pickle.dump(token2word, f, -1)
