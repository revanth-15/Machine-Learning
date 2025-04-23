class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def save_weights(self, filepath):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias'):
                weights[f'layer_{i}_bias'] = layer.bias
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if f'layer_{i}_weights' in weights:
                layer.weights = weights[f'layer_{i}_weights']
            if f'layer_{i}_bias' in weights:
                layer.bias = weights[f'layer_{i}_bias']