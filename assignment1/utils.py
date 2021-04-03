import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def normalize(data, mean, std):
    return (data - mean) / std


def one_hot_encode_labels(labels):
    vector = np.array(labels)
    one_hot = np.zeros((vector.size, vector.max() + 1))
    one_hot[np.arange(vector.size), vector] = 1
    return one_hot


def train_validation_error(train_errors, validation_errors, title='', save=None):
    plt.clf()
    plt.plot(train_errors, label='train error')
    plt.plot(validation_errors, label='validation error')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    if save is None:
        plt.show()
    else:
        plt.savefig("result_pics/error_" + save)


def montage(W, labels, title='', save=None):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2, 5)
    fig.suptitle(title)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title(labels[5 * i + j])
            ax[i][j].axis('off')
    if save is None:
        plt.show()
    else:
        plt.savefig("result_pics/montage_" + save)


def show_image(image):
    """ Display an image from the dataset """
    import matplotlib.pyplot as plt
    # assert image.nsize == 32*32*3, "image size not compatible"
    image = image.reshape(32, 32, 3, order='F')
    sim = (image - np.min(image[:])) / (np.max(image[:]) - np.min(image[:]))
    sim = sim.transpose(1, 0, 2)
    plt.imshow(sim, interpolation='nearest')
    plt.show()

