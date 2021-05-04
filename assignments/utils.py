import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def normalize(data, mean, std):
    return (data - mean) / std


def train_validation_error(train_errors, validation_errors, title='', save=None, label='', steps=0):
    data_points = len(train_errors)
    x_axis = [(i + 1) * steps/data_points for i in range(data_points)]

    plt.clf()
    plt.plot(x_axis, train_errors, label='training')
    plt.plot(x_axis, validation_errors, label='validation')
    plt.title(title)
    plt.xlabel('update step')
    plt.ylabel(label)
    plt.legend()
    if save is None:
        plt.show()
    else:
        plt.savefig("result_pics/" + label + "-" + save)


def compare_batch_plot(train_errors, validation_errors, batch_train_errors, batch_val_errors
                       , title='', save=None, label='', steps=0, comparing_label='bn'):
    data_points = len(train_errors)
    x_axis = [(i + 1) * steps/data_points for i in range(data_points)]

    plt.clf()
    plt.plot(x_axis, train_errors, 'c', label=f'training without {comparing_label}')
    plt.plot(x_axis, validation_errors, 'c--', label=f'validation without {comparing_label}')
    plt.plot(x_axis, batch_train_errors, 'b', label=f'training with {comparing_label}')
    plt.plot(x_axis, batch_val_errors, 'b--', label=f'validation with {comparing_label}')
    plt.title(title)
    plt.xlabel('update step')
    plt.ylabel(label)
    plt.legend()
    if save is None:
        plt.show()
    else:
        plt.savefig("result_pics/" + label + "-" + save)


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
        plt.savefig("result_pics/montage-" + save)


def show_image(image):
    """ Display an image from the dataset """
    import matplotlib.pyplot as plt
    # assert image.nsize == 32*32*3, "image size not compatible"
    image = image.reshape(32, 32, 3, order='F')
    sim = (image - np.min(image[:])) / (np.max(image[:]) - np.min(image[:]))
    sim = sim.transpose(1, 0, 2)
    plt.imshow(sim, interpolation='nearest')
    plt.show()


def translate(images, shift=10, vertical=False):
    if vertical:
        images = images.copy()
        right_slice = images[-shift:, :].copy()
        images[shift:, :] = images[:-shift, :]
        images[:shift, :] = np.fliplr(right_slice)
        return images
    else:
        images = images.copy()
        right_slice = images[:, -shift:].copy()
        images[:, shift:] = images[:, :-shift]
        images[:, :shift] = np.fliplr(right_slice)
        return images


def horizontal_flip(images):
    size = images.shape[0]
    size /= 3
    size = int(np.sqrt(size))
    for i in range(3 * size):
        a = images[:i * size, :]
        b = np.flipud(images[i * size:(i + 1) * size, :])
        c = images[(i + 1) * size:, :]
        images = np.concatenate((a, b, c))
    return images
