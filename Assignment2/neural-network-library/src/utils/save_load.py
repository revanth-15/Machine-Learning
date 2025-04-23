def save_weights(model, file_path):
    weights = model.get_weights()
    with open(file_path, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(model, file_path):
    with open(file_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)