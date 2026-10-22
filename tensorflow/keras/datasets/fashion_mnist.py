import gzip
import os
import urllib.request
import numpy as np

def load_data():
    """
    Loads the Fashion-MNIST dataset.
    Returns:
        Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    cache_dir = os.path.expanduser(os.path.join('~', '.keras', 'datasets', 'fashion-mnist'))
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for fname in files:
        fpath = os.path.join(cache_dir, fname)
        if not os.path.exists(fpath):
            req = urllib.request.Request(base_url + fname, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(fpath, 'wb') as out:
                out.write(resp.read())
        paths.append(fpath)

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        X_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        X_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (X_train, y_train), (X_test, y_test)
