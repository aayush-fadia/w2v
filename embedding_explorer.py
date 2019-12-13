import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as tf
from DataSupplier import DataSupplier
from mpl_toolkits import mplot3d
import pandas as pd
import warnings

warnings.simplefilter('ignore', FutureWarning)
fig = plt.figure()
ax = fig.gca(projection='3d')
ds = DataSupplier()
words = 'nice great good terrible horrible bad'
words_for_display_text = words.strip().split(' ')

words_for_display = [ds.word2token[word] for word in words_for_display_text]

pca = PCA(n_components=3)
def plot_words_pca(model):
    print(model.layers)
    w = (model.layers[0].get_weights()[0] + model.layers[1].get_weights()[0].T) / 2
    df = pd.DataFrame(w)
    pca_results = pca.fit_transform(StandardScaler().fit_transform(df.values[words_for_display, :]))
    ax.cla()
    ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2])
    for j in range(len(pca_results)):
        ax.text(pca_results[j, 0], pca_results[j, 1], pca_results[j, 2], words_for_display_text[j])
    fig.show()
    plt.pause(1000000)

def get_model():
    RUN_NAME = input('Enter Run Name')
    print('Loading Saved Model')
    model = tf.models.load_model('scratchspace/model_{}.h5'.format(RUN_NAME))
    RUN_NAME = RUN_NAME + 'run_2'
    return model
mdl = get_model()
plot_words_pca(mdl)
