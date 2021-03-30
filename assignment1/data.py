def load_batch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('datasets/' + filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


def dict_to_data_and_label(data):
    return data[b'data'], data[b'labels']


def load_label_names():
    import pickle
    with open('datasets/batches.meta', 'rb') as fo:
        names = pickle.load(fo, encoding='bytes')
    return names[b'label_names']
