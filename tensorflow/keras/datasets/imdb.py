import os
import json
import numpy as np

def get_word_index(path='imdb_word_index.json'):
    cache_path = os.path.expanduser(os.path.join('~', '.keras', 'datasets', 'imdb_word_index.json'))
    if not os.path.exists(cache_path):
        cache_path = os.path.join(r'C:\Users\baps\.keras\datasets', 'imdb_word_index.json')
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(path='imdb.npz', num_words=None, skip_top=0, maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3):
    cache_path = os.path.expanduser(os.path.join('~', '.keras', 'datasets', 'imdb.npz'))
    if not os.path.exists(cache_path):
        cache_path = os.path.join(r'C:\Users\baps\.keras\datasets', 'imdb.npz')
        
    data = np.load(cache_path, allow_pickle=True)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    # Reproducible shuffle
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices_test = np.arange(len(x_test))
    rng.shuffle(indices_test)
    x_test = x_test[indices_test]
    y_test = y_test[indices_test]

    def _process_sequence(seq):
        if skip_top > 0:
            seq = [w for w in seq if w >= skip_top]
        
        # shift indices
        if index_from:
            seq = [w + index_from for w in seq]
            
        if start_char is not None:
            seq = [start_char] + seq
            
        if num_words is not None:
            seq = [w if w < num_words else oov_char for w in seq]
            
        return seq

    x_train = np.array([_process_sequence(s) for s in x_train], dtype=object)
    x_test = np.array([_process_sequence(s) for s in x_test], dtype=object)

    if maxlen:
        new_x_train, new_y_train = [], []
        for x, y in zip(x_train, y_train):
            if len(x) < maxlen:
                new_x_train.append(x)
                new_y_train.append(y)
        x_train, y_train = np.array(new_x_train, dtype=object), np.array(new_y_train)

        new_x_test, new_y_test = [], []
        for x, y in zip(x_test, y_test):
            if len(x) < maxlen:
                new_x_test.append(x)
                new_y_test.append(y)
        x_test, y_test = np.array(new_x_test, dtype=object), np.array(new_y_test)

    return (x_train, y_train), (x_test, y_test)
