from time import clock

from PizzaCutter import *


def read_file(filename):
    with open(filename) as f:
        R, C, L, H = [int(i) for i in f.readline().split()]
        pizza_matrix = np.zeros((R, C), dtype=np.int8)  # 0 for tomato, 1 for mushroom

        for row in range(R):
            for col, char in enumerate(f.readline().strip('\n')):
                if char == 'M':
                    pizza_matrix[row][col] = 1

    return L, H, pizza_matrix


def write_file(filename, L):
    with open(filename, 'w') as f:
        f.write(str(len(L)) + '\n')
        for x in L:
            f.write(' '.join(str(i) for i in x) + '\n')


def score(X):
    '''
    Computes the score of a given solution.
    
    :param X: List of slices, each slice is a tuple (x0,y0,x1,y1)
    :return: sum over all cells covered by the slices in X
    '''
    sc = 0
    for x0, y0, x1, y1 in X:
        sc+= (x1 - x0 + 1) * (y1 - y0 + 1)
    return sc


if __name__ == '__main__':
    while True:
        filename = input()
        L, H, mat = read_file(filename + '.in')
        start = clock()
        C = PizzaCutter(L, H, mat)
        X = C.slice_pizza()
        end = clock()
        sc = score(X)
        write_file(filename + '.ot', X)
        print('score:', sc)
        print('time: {} seconds'.format(end - start))
