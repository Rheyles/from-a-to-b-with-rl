import matplotlib.pyplot as plt
import sys


def plot_loss(losses, show_result=False):
    """ Plots the loss function as a function of time"""

    plt.ion()
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(losses)

    plt.pause(0.001)  # pause a bit so that plots are updated


def pretty_print(arrays : tuple, transpose=False, names=None):
    ''' Brice wants pretty prints of arrays
    or tuples of arrays. The function is a bit ugly, but it works.
    '''

    n_rows = 0 
    if names is None: 
        names = [f'Table {no}' for no in range(len(arrays))]

    for array, name in zip(arrays, names):
        if transpose: array = np.array(array).T

        n_char_table = 7 * array.shape[1]
        n_char_name = len(name)
        left_pad = '-'*((n_char_table - n_char_name)//2)
        right_pad = '-'*((1 + n_char_table - n_char_name)//2)

        pretty_str = left_pad + name + right_pad + '\n'

        for row in array:
            n_rows += 1
            pretty_str += ''.join([f'{val:.2f} | '  for val in row])
            pretty_str += '\n'
        pretty_str += '-' *  n_char_table + '\n'
        n_rows += 2

        print(pretty_str, end='')

    for row in range(n_rows+1):
        sys.stdout.write("\033[F") # Cursor up one line


if __name__ == '__main__':
    import numpy as np

    x = np.random.uniform(size=(4,10))
    y = np.random.uniform(size=(4,10))
    pretty_print((x,y), names=('fake x', 'real x'))

