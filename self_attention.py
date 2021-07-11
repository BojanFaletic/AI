import numpy as np


def softmax(x: np.array) -> np.array:
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def ReLU(x: np.array) -> np.array:
    zero = np.zeros_like(x)
    return np.maximum(zero, x)


class Self_attention:
    def __init__(self, n_size=(4, 3)):
        self.key_weight = np.random.uniform(size=n_size)
        self.query_weight = np.random.uniform(size=n_size)
        self.value_weight = np.random.uniform(size=n_size)

    def forward(self, input_full):
        # this are just linear layers
        key = input_full @ self.key_weight
        value = input_full @ self.value_weight
        query = input_full @ self.query_weight

        # calculate attention score (d leads to more stable gradients)
        d = np.sqrt(np.sum(key.shape))
        attention = query @ key.T / d

        # normalize with softmax
        attention_norm = softmax(attention).T

        # final output calculated from attention and values
        output = attention_norm @ value
        return output

    def __call__(self, x):
        return self.forward(x)


class Multihead_attention:
    def __init__(self, head_cnt=2, n_size=(4, 3)):
        # just crate n-attention networks
        self.attention = [Self_attention(n_size) for i in range(head_cnt)]

        # weight matrix, to aggregate n-heads
        weight_shape = (n_size[1]*head_cnt, n_size[0])
        self.weight_matrix = np.random.uniform(size=(weight_shape))

        self.head_cnt = head_cnt
        self.n_size = n_size

    def forward(self, x):
        # create output with shape (head, n_input, input_len)
        h, _, n_len = (self.head_cnt, *self.n_size)
        z_out = np.zeros((h*n_len, n_len))

        # concatenate output from each attention network into
        for i, head in enumerate(self.attention):
            z_out[i*n_len:(i+1)*n_len] = head(x)

        # multiply self attention with weight matrix
        output = z_out.T @ self.weight_matrix
        return output

    def __call__(self, x):
        return self.forward(x)


class LayerNorm:
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.beta = np.random.rand()
        self.gamma = np.random.rand()

    def forward(self, x):
        mean = np.mean(x)
        var = np.var(x)
        y = (x - mean) / np.sqrt(var + self.eps) * self.gamma + self.beta
        return y

    def __call__(self, x):
        return self.forward(x)


class Encoder_layer:
    def __init__(self, head_cnt=2, n_size=(4, 3)):
        self.multihead_attention = Multihead_attention(head_cnt, n_size)

        # layer norm parameters
        self.layer_norm_1 = LayerNorm()
        self.layer_norm_2 = LayerNorm()

        # feed forward param (2 layer)
        self.weight_1 = np.random.uniform(size=(n_size[0], n_size[0]))
        self.bias_1 = np.random.uniform(size=(n_size)).T

        self.weight_2 = np.random.uniform(size=(n_size[0], n_size[0]))
        self.bias_2 = np.random.uniform(size=(n_size)).T

    def forward(self, x):
        out = self.multihead_attention(x)

        # Add + LayerNorm
        out = out + x
        y = self.layer_norm_1(out)

        # Feed forward (1 layers)
        out = y @ self.weight_1 + self.bias_1

        out = ReLU(out)
        # Feed forward (2 layer)
        out = out @ self.weight_2 + self.bias_2

        # Add + LayerNorm
        out = y + out
        y = self.layer_norm_2(out)

        return y

    def __call__(self, x):
        return self.forward(x)


class Encoder:
    def __init__(self, enc_blk=1, head_cnt=2, n_size=(4, 3)):
        self.enc_layer = [Encoder_layer(head_cnt, n_size)
                          for _ in range(enc_blk)]

    def forward(self, x):
        # encoder is just n-stacks of encoder-layer
        out = x
        for encoder in self.enc_layer:
            out = encoder(out)
        return out

    def __call__(self, x):
        return self.forward(x)


# test
if __name__ == "__main__":
    att = Self_attention((4, 3))
    input_full = np.random.uniform(size=(3, 4))

    # Encoder architecture like BERT
    en = Encoder(enc_blk=10)
    en(input_full)
