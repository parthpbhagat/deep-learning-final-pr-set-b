import numpy as np

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
    """Pads sequences to the same length."""
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    
    num_samples = len(sequences)
    if num_samples == 0:
        return np.empty((0, maxlen or 0), dtype=dtype)
        
    lengths = [len(s) for s in sequences]
    if maxlen is None:
        maxlen = int(np.max(lengths))
        
    x = np.full((num_samples, maxlen), value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f"Truncating type '{truncating}' not understood")
            
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f"Padding type '{padding}' not understood")
    return x
