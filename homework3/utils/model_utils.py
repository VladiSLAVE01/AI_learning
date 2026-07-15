def make_layer_config(hidden_sizes, use_batch_norm=False, activation='relu', dropout_p=0.5, l2_reg=False):
    layers = []

    for size in hidden_sizes:
        layers.append({'type': 'linear', 'size': size})
        if use_batch_norm:
            layers.append({'type': 'batch_norm'})
        if activation:
            layers.append({'type': activation})
        if dropout_p > 0:
            layers.append({'type': 'dropout', 'rate': dropout_p})

    return layers
