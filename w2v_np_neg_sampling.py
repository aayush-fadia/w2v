import numpy as np
from DataSupplier import DataSupplier
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

plt.ion()
ds = DataSupplier(5, 15)
N_WORDS = ds.N_WORDS
N_DIM = 150
LR = 0.1
LR_DECAY = 0.9990
word_vectors = np.random.randn(N_WORDS * N_DIM).reshape((N_WORDS, N_DIM)) / 10000
context_vectors = np.random.randn(N_WORDS * N_DIM).reshape((N_WORDS, N_DIM)) / 10000


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def sim(x, y):
    dt = np.dot(x, y.T)
    return sigmoid(dt)


fig, ax = plt.subplots()
words_for_display = [ds.word2token[word] for word in ['great', 'good', 'bad', 'horrible', 'terrible', 'nice']]
pca = PCA(n_components=2)

for i in range(1000000000):
    LR *= LR_DECAY
    print(LR)
    w, s, o = ds.getExampleFullContext(random.randint(0, N_WORDS - 1))
    Wx = word_vectors[w]
    Cs = [context_vectors[i] for i in s]
    Co = [context_vectors[i] for i in o]
    WxCs = [1 - sim(Wx, cs) for cs in Cs]
    WxCo = [1 - sim(Wx, -co) for co in Co]
    WxCsCs = [WxCs[i] * Cs[i] for i in range(len(s))]
    WxCoCo = [WxCo[i] * Co[i] for i in range(len(o))]
    Wx = Wx + LR * (sum(WxCsCs) - sum(WxCoCo))
    word_vectors[w] = Wx
    WxCsWx = [wxcs * Wx for wxcs in WxCs]
    for i in range(len(s)):
        context_vectors[s[i]] = context_vectors[s[i]] + LR * WxCsWx[i]
    WxCoWx = [wxco * Wx for wxco in WxCo]
    for i in range(len(o)):
        context_vectors[o[i]] = context_vectors[o[i]] - LR * WxCoWx[i]
    df = pd.DataFrame(word_vectors)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(df.values[words_for_display, :]))
    tsne_pca_results = pca_result
    ax.cla()
    ax.scatter(tsne_pca_results[:, 0], tsne_pca_results[:, 1])
    for j in range(len(pca_result)):
        ax.annotate(ds.word(words_for_display[j]), (tsne_pca_results[j, 0], tsne_pca_results[j, 1]))
    plt.show()
    plt.pause(0.00001)
