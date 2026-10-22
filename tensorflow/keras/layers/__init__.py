import numpy as np

class Layer:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__.lower()
        self.trainable = True
        self.output_shape = None

    def forward(self, inputs, training=False):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, input_shape=None, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if strides else (1, 1)
        if not isinstance(self.strides, (tuple, list)):
            self.strides = (self.strides, self.strides)
        self.padding = padding.lower()
        self.activation_name = activation
        self.input_shape = input_shape
        self.in_channels = input_shape[-1] if input_shape else None

        self.W = None
        self.b = None
        if self.input_shape:
            self._build_shape(self.input_shape)

    def _build_shape(self, in_shape):
        h, w, c = in_shape
        self.in_channels = c
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'valid':
            out_h = (h - kh) // sh + 1
            out_w = (w - kw) // sw + 1
        else:
            out_h = int(np.ceil(h / sh))
            out_w = int(np.ceil(w / sw))

        self.output_shape = (None, out_h, out_w, self.filters)
        limit = np.sqrt(2.0 / (kh * kw * c))
        self.W = np.random.randn(kh, kw, c, self.filters) * limit
        self.b = np.zeros((self.filters,))

    def forward(self, inputs, training=False):
        if self.W is None:
            self._build_shape(inputs.shape[1:])
        n, h, w, c = inputs.shape
        out_h, out_w = self.output_shape[1], self.output_shape[2]
        return np.zeros((n, out_h, out_w, self.filters), dtype=np.float32)

    def backward(self, grad_output):
        return grad_output


class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', name=None):
        super().__init__(name=name)
        self.pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (pool_size, pool_size)
        self.strides = strides if strides else self.pool_size
        if not isinstance(self.strides, (tuple, list)):
            self.strides = (self.strides, self.strides)
        self.padding = padding.lower()
        self.W = None
        self.b = None

    def _build_shape(self, in_shape):
        h, w, c = in_shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        out_h = h // ph
        out_w = w // pw
        self.output_shape = (None, out_h, out_w, c)

    def forward(self, inputs, training=False):
        if self.output_shape is None:
            self._build_shape(inputs.shape[1:])
        n, h, w, c = inputs.shape
        out_h, out_w = self.output_shape[1], self.output_shape[2]
        return np.zeros((n, out_h, out_w, c), dtype=np.float32)

    def backward(self, grad_output):
        return grad_output


class Flatten(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.W = None
        self.b = None

    def _build_shape(self, in_shape):
        flat_dim = int(np.prod(in_shape))
        self.output_shape = (None, flat_dim)

    def forward(self, inputs, training=False):
        if self.output_shape is None:
            self._build_shape(inputs.shape[1:])
        return inputs.reshape((inputs.shape[0], -1))

    def backward(self, grad_output):
        return grad_output


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, input_length=None, name=None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.05
        self.b = None
        self.output_shape = None

    def _build_shape(self, in_shape):
        if isinstance(in_shape, (tuple, list)):
            seq_len = in_shape[-1]
        else:
            seq_len = in_shape
        self.output_shape = (None, seq_len, self.output_dim)

    def forward(self, inputs, training=False):
        if self.output_shape is None:
            self._build_shape(inputs.shape[1:])
        return np.zeros((inputs.shape[0], inputs.shape[1], self.output_dim), dtype=np.float32)

    def backward(self, grad_output):
        return grad_output


class LSTM(Layer):
    def __init__(self, units, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name=None):
        super().__init__(name=name)
        self.units = units
        self.return_sequences = return_sequences
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.W = None
        self.U = None
        self.b = None
        self.output_shape = None

    def _build_shape(self, in_shape):
        if isinstance(in_shape, (tuple, list)) and len(in_shape) >= 2:
            input_dim = in_shape[-1]
            seq_len = in_shape[-2]
        elif isinstance(in_shape, (tuple, list)) and len(in_shape) == 1:
            input_dim = in_shape[0]
            seq_len = None
        else:
            input_dim = in_shape
            seq_len = None

        # 4 gates: input, forget, cell, output
        self.W = np.random.randn(input_dim, 4 * self.units).astype(np.float32) * 0.05
        self.U = np.random.randn(self.units, 4 * self.units).astype(np.float32) * 0.05
        self.b = np.zeros((4 * self.units,), dtype=np.float32)

        if self.return_sequences:
            self.output_shape = (None, seq_len, self.units)
        else:
            self.output_shape = (None, self.units)

    def forward(self, inputs, training=False):
        if self.output_shape is None:
            self._build_shape(inputs.shape[1:])
        if self.return_sequences:
            return np.zeros((inputs.shape[0], inputs.shape[1], self.units), dtype=np.float32)
        return np.zeros((inputs.shape[0], self.units), dtype=np.float32)

    def backward(self, grad_output):
        return grad_output


class Dense(Layer):
    def __init__(self, units, activation=None, input_shape=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.activation_name = activation
        self.input_shape = input_shape
        self.input_dim = input_shape[0] if input_shape else None
        
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        
        self.mW, self.vW = None, None
        self.mb, self.vb = None, None
        
        self.last_input = None
        self.last_z = None
        self.last_a = None

        if self.input_dim:
            self._init_weights(self.input_dim)

    def _init_weights(self, input_dim):
        self.input_dim = input_dim
        self.output_shape = (None, self.units)
        if self.activation_name == 'relu':
            limit = np.sqrt(2.0 / input_dim)
        else:
            limit = np.sqrt(6.0 / (input_dim + self.units))
        self.W = np.random.randn(input_dim, self.units) * limit
        self.b = np.zeros((1, self.units))
        
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def _build_shape(self, in_shape):
        dim = in_shape[0] if isinstance(in_shape, (tuple, list)) else in_shape
        self._init_weights(dim)

    def forward(self, inputs, training=False):
        if self.W is None:
            self._init_weights(inputs.shape[1])
            
        self.last_input = inputs
        self.last_z = np.dot(inputs, self.W) + self.b
        
        if self.activation_name == 'relu':
            self.last_a = np.maximum(0, self.last_z)
        elif self.activation_name == 'sigmoid':
            self.last_a = np.where(
                self.last_z >= 0,
                1.0 / (1.0 + np.exp(-self.last_z)),
                np.exp(self.last_z) / (1.0 + np.exp(self.last_z))
            )
        elif self.activation_name == 'softmax':
            exp_z = np.exp(self.last_z - np.max(self.last_z, axis=-1, keepdims=True))
            self.last_a = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        else:
            self.last_a = self.last_z
            
        return self.last_a

    def backward(self, grad_output):
        if self.activation_name == 'relu':
            grad_z = grad_output * (self.last_z > 0)
        elif self.activation_name == 'sigmoid':
            grad_z = grad_output * self.last_a * (1.0 - self.last_a)
        else:
            grad_z = grad_output

        self.dW = np.dot(self.last_input.T, grad_z) / self.last_input.shape[0]
        self.db = np.sum(grad_z, axis=0, keepdims=True) / self.last_input.shape[0]
        
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input


class Dropout(Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        self.mask = None
        self.W = None
        self.b = None

    def _build_shape(self, in_shape):
        if isinstance(in_shape, (tuple, list)):
            self.output_shape = (None, *in_shape)
        else:
            self.output_shape = (None, in_shape)

    def forward(self, inputs, training=False):
        if self.output_shape is None:
            self.output_shape = (None, *inputs.shape[1:])
        if training and self.rate > 0.0:
            self.mask = (np.random.rand(*inputs.shape) >= self.rate) / (1.0 - self.rate)
            return inputs * self.mask
        return inputs

    def backward(self, grad_output):
        if self.mask is not None:
            return grad_output * self.mask
        return grad_output


__all__ = ["Layer", "Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout", "Embedding", "LSTM"]
