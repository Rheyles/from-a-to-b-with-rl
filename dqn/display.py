import matplotlib.pyplot as plt
import numpy as np
import typing
import sys


class Plotter():

    plt.ion()
    fig_data_list = []

    @classmethod
    def plot_data_gradually(cls, data_name: str, data: list[float,int],
                            show_result: bool=False) -> None:
        """
        Plots gradually the data as a function of time. Each data is plotted in
        its own figure
        ex : Plotter().plot_data_gradually(data,data_name)

        Args:
            data_name (str): the name of the data you want to plot
            data (list[float,int]): the data you want to plot
            show_result (bool, optional): true if the data won't change anymore.
                                          Defaults to False.
        """

        # We get the correct Figure object
        if data_name not in cls.fig_data_list:
            cls.fig_data_list.append(data_name)
            plt.figure(len(cls.fig_data_list))
        else:
            plt.figure(cls.fig_data_list.index(data_name) + 1)

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Step')
        plt.ylabel(data_name)
        plt.plot(data)

        plt.pause(0.001)  # pause a bit so that plots are updated

        if show_result:
            plt.show()


def pretty_print(arrays : tuple, transpose=False, names=None):
    ''' Brice wants pretty prints of arrays
    or tuples of arrays. The function is a bit ugly, but it works.
    '''

    n_rows = 0
    if names is None:
        names = [f'Table {no}' for no in range(len(arrays))]

    for array, name in zip(arrays, names):
        array = np.atleast_2d(array)
        if transpose: array = array.T

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

    # for row in range(n_rows+1):
        # sys.stdout.write("\033[F") # Cursor up one line


if __name__ == '__main__':
    # import numpy as np

    # x = np.random.uniform(size=(4,10))
    # y = np.random.uniform(size=(4,10))
    # pretty_print((x,y), names=('fake x', 'real x'))

    data = []
    data_2 = []
    for i in range(50):
        data.append(i**2)
        data_2.append(i+1)
        Plotter().plot_data_gradually('Test', data)
        Plotter().plot_data_gradually('Toto', data_2)
    data.append(i**2)
    Plotter().plot_data_gradually('Test', data, show_result=True)
    Plotter().plot_data_gradually('Toto', data_2, show_result=True)
