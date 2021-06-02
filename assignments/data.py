import numpy as np
from utils import show_image


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


def load_data(file_name):
    train_data = load_batch(file_name)
    X, Y = dict_to_data_and_label(train_data)
    Y = one_hot_encode_labels(Y)
    y = [image.decode('utf-8') for image in load_label_names()]
    return X.T, Y.T, y


def load_all_data(validation=False):
    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    X, Y = None, None
    for file in files:
        train_data = load_batch(file)
        x_batch, y_batch = dict_to_data_and_label(train_data)
        y_batch = one_hot_encode_labels(y_batch)

        if validation:
            x_batch = x_batch[9800:, :]
            y_batch = y_batch[9800:, :]
        else:
            x_batch = x_batch[:9800, :]
            y_batch = y_batch[:9800, :]

        if X is None:
            X = x_batch.T
            Y = y_batch.T
        else:
            X = np.concatenate((X, x_batch.T), axis=1)
            Y = np.concatenate((Y, y_batch.T), axis=1)

    y = [image.decode('utf-8') for image in load_label_names()]
    return X, Y, y


def one_hot_encode_labels(labels):
    vector = np.array(labels)
    one_hot = np.zeros((vector.size, vector.max() + 1))
    one_hot[np.arange(vector.size), vector] = 1
    return one_hot


def explore_images(n, input_data):
    for i in range(n):
        show_image(input_data[:, i])


def read_text_file(filename):
    with open('datasets/' + filename, 'r') as f:
        text = f.read()
    return text


def load_and_process_text(filename):
    book_data = [c for c in read_text_file(filename)]
    book_chars = list(set(book_data))
    char_2_indices = {val: index for index, val in enumerate(book_chars)}
    indices_2_char = {index: val for index, val in enumerate(book_chars)}
    return {'book_data': book_data,
            'book_chars': book_chars,
            'char_2_indices': char_2_indices,
            'indices_2_char': indices_2_char}
