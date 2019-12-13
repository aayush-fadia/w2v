import tensorflow.keras as tf
from DataSupplier import DataSupplier
import matplotlib.pyplot as plt

plt.ion()
from mpl_toolkits import mplot3d
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import tracemalloc
tracemalloc.start()
warnings.simplefilter('ignore', FutureWarning)

BATCH_SIZE = 8
N_DIM = 50
N_EPOCHS = 2
ds = DataSupplier(4, 1)
N_WORDS = ds.N_WORDS
print(N_WORDS)

RUN_NAME = input('Give this run a name:')
fig = plt.figure()
ax = fig.gca(projection='3d')

words_for_display_text = ['great', 'good', 'bad', 'horrible', 'terrible', 'nice']

words_for_display = [ds.word2token[word] for word in
                     ['great', 'good', 'bad', 'horrible', 'terrible', 'nice']]
pca = PCA(n_components=3)
log_dir = "scratchspace/tblogs/fit/{}".format(RUN_NAME)
tensorboard_callback = tf.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq='batch',
                                                write_graph=False, write_images=False)


def get_model():
    global RUN_NAME
    if 'model_{}.h5'.format(RUN_NAME) not in os.listdir('scratchspace/'):
        print("Building Model")
        model = tf.Sequential()
        model.add(tf.layers.Dense(N_DIM, input_dim=N_WORDS, name='word_vectors', use_bias=False))
        model.add(tf.layers.Dense(N_WORDS, activation=tf.activations.softmax, use_bias=False))
        model.compile(tf.optimizers.Adam(learning_rate=0.00001), tf.losses.categorical_crossentropy)
        os.mkdir('scratchspace/figs_{}'.format(RUN_NAME))
        return model
    else:
        print('Loading Saved Model')
        model = tf.models.load_model('scratchspace/model_{}.h5'.format(RUN_NAME))
        RUN_NAME = RUN_NAME + '+'
        os.mkdir('scratchspace/figs_{}'.format(RUN_NAME))
        return model


def plot_words_pca(model):
    print(model.layers)
    w = (model.layers[0].get_weights()[0] + model.layers[1].get_weights()[0].T) / 2
    df = pd.DataFrame(w)
    pca_results = pca.fit_transform(StandardScaler().fit_transform(df.values[words_for_display, :]))
    ax.cla()
    ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2])
    for j in range(len(pca_results)):
        ax.text(pca_results[j, 0], pca_results[j, 1], pca_results[j, 2], words_for_display_text[j])
    fig.savefig('scratchspace/figs_{}/projection_step{}.png'.format(RUN_NAME, i))
    fig.show()
    plt.pause(0.1)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

model = get_model()
i = 0
while True:
    ws, cs, _ = ds.getBatch(BATCH_SIZE * 32)
    ws = tf.utils.to_categorical(ws, N_WORDS)
    cs = tf.utils.to_categorical(cs, N_WORDS)
    model.fit(ws, cs, BATCH_SIZE, epochs=N_EPOCHS, callbacks=[tensorboard_callback])
    if i % 15 == 0:
        plot_words_pca(model)
    if i % 50 == 0:
        model.save('scratchspace/model_{}.h5'.format(RUN_NAME))
    i += 1
