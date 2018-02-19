import numpy as np


class PizzaCutter():
    def __init__(self, L, H, mat):
        '''
        Initializes a PizzaCutter Object and computes the add_matrix. add_matrix is a matrix where the (i,j) position
        contains a tuple (t,m) with t and m being the number of tomatoes and mushrooms in the slice  given by the two
        coordinates (0,0) and (i,j). This allows to compute the number of tomatoes and mushrooms in any given slice
        (x0,y0)--(x1,y1) in constant time.
        Furthermore a 0-1-matrix cover_matrix is initialized. At the (i,j) position we have 1 if and only if the cell
        (i,j) is covered by some slice.

        :param L: minimum number of each ingredient per slice
        :param H: maximum number of cells per slice
        :param mat: 0-1-matrix representing the given pizza. 0: tomato, 1:mushroom
        '''
        self.L, self.H, self.matrix = L, H, mat
        self.height, self.width = self.matrix.shape
        self.count = 0

        # Computation of add_matrix using dynamic programming
        self.add_matrix = np.zeros((self.height, self.width), dtype=(int, 2))
        self.cover_matrix = np.zeros((self.height, self.width))
        self.add_matrix[0][0] = (1, 0) if self.matrix[0][0] == 0 else (0, 1)
        for row in range(self.height):
            for col in range(self.width):
                left = self.add_matrix[row][col - 1] if col - 1 >= 0 else (0, 0)
                top = self.add_matrix[row - 1][col] if row - 1 >= 0 else (0, 0)
                topleft = self.add_matrix[row - 1][col - 1] if row - 1 >= 0 and col - 1 >= 0 else (0, 0)
                x = (0, 1) if self.matrix[row][col] == 1 else (1, 0)
                self.add_matrix[row][col] = (left[0] + top[0] - topleft[0] + x[0], left[1] + top[1] - topleft[1] + x[1])

    @staticmethod
    def order_cords(x0, x1, y0, y1):
        '''
        Sorts coordinates in an increasing order.

        :return: Tuple (x0,x1,y0,y1) where x0<x1 and y0<y1
        '''

        if x0 > x1:
            x0, x1 = x1, x0

        if y0 > y1:
            y0, y1 = y1, y0

        return x0, x1, y0, y1

    def number_of_ingredients(self, x0, x1, y0, y1):
        '''
        Computes the number of ingredients in the rectangle between (x0,y0) and (x1,y1).
        Has a constant running time. the coordinates (x0,x1) and (y0,y1) can be given in any order.

        :return: tuple (t,m) where t is the number of tomatoes and m is the number of mushrooms
        '''

        x0, x1, y0, y1 = self.order_cords(x0, x1, y0, y1)

        left = self.add_matrix[x0 - 1][y1] if x0 > 0 else (0, 0)
        top = self.add_matrix[x1][y0 - 1] if y0 > 0 else (0, 0)
        topleft = self.add_matrix[x0 - 1][y0 - 1] if x0 > 0 and y0 > 0 else (0, 0)

        return tuple([self.add_matrix[x1][y1][i] - left[i] - top[i] + topleft[i] for i in range(2)])

    def ratio(self, x0, y0, x1, y1):
        '''
        Computes the ratio of tomatos in a given slice

        :return: Float - #tomatos/(size of the slice) in the slice given by the coordinates (x0,y0)--(x1,y1)
        '''
        x0, x1, y0, y1 = self.order_cords(x0, x1, y0, y1)

        t, m = self.number_of_ingredients(x0, y0, x1, y1)
        return t / (t + m)

    def has_feasible_ingred(self, x0, x1, y0, y1):
        '''
        Determines whether a slice has enough of each ingredient.

        :return: True, iff #tomatos and #shrooms is bigger than L in the slice (x0,y0)--(x1,y1)
        '''
        t, m = self.number_of_ingredients(x0, x1, y0, y1)
        return t >= self.L and m >= self.L

    def find_optimal_cut(self, x0, x1, y0, y1):
        '''
        Computes a single optimal cut for a given rectangle. Here optimal means that both pieces have similar ratios of
        Shrooms to Tomatos and both slices have enough ingredients (more than L).


        :return: Tuple of direction (either h (horizontal) or v (vertical)) and the position p of the cut. 'n',-1 if
        no cut fulfilling the L-condition exists.
        '''

        x0, x1, y0, y1 = self.order_cords(x0, x1, y0, y1)

        opt_vert_cut, opt_hor_cut = [], []

        ratio_diff_vert = ratio_diff_hor = 2

        for vert_pos in range(x0, x1): #Compute feasible vertical cuts
            if self.has_feasible_ingred(x0, vert_pos, y0, y1) and self.has_feasible_ingred(vert_pos + 1, x1, y0, y1):
                opt_vert_cut.append(vert_pos)
                ratio_left = self.ratio(x0, vert_pos, y0, y1)
                ratio_right = self.ratio(vert_pos + 1, x1, y0, y1)
                if abs(ratio_left - ratio_right) < ratio_diff_vert - 0.08:
                    ratio_diff_vert = abs(ratio_left - ratio_right)
                    opt_vert_cut = [vert_pos]
                elif ratio_diff_vert - 0.08 <= abs(ratio_left - ratio_right) <= ratio_diff_vert + 0.08:
                    opt_vert_cut.append(vert_pos)

        for hor_pos in range(y0, y1): #Compute feasible horizontal cuts
            if self.has_feasible_ingred(x0, x1, y0, hor_pos) and self.has_feasible_ingred(x0, x1, hor_pos + 1, y1):
                opt_hor_cut.append(hor_pos)

                ratio_upper = self.ratio(x0, x1, y0, hor_pos)
                ratio_lower = self.ratio(x0, x1, hor_pos + 1, y1)
                if abs(ratio_upper - ratio_lower) < ratio_diff_hor - 0.08:
                    ratio_diff_hor = abs(ratio_upper - ratio_lower)
                    opt_hor_cut = [hor_pos]
                elif ratio_diff_hor + 0.08 <= abs(ratio_upper - ratio_lower) <= ratio_diff_hor + 0.08:
                    opt_hor_cut.append(hor_pos)

        if opt_vert_cut == opt_hor_cut == []: #no feasible cut exists
            return 'n', -1
        elif ratio_diff_vert <= ratio_diff_hor:
            return 'v', min(opt_vert_cut, key=lambda i: abs((x1 - x0) // 2 - i))
        else:
            return 'h', min(opt_hor_cut, key=lambda i: abs((y1 - y0) // 2 - i))

    def is_slice(self, x0, x1, y0, y1):
        '''
        Checks whether a given slice matches the L and H conditions.

        :return: True, iff the slice (x0,y0)--(x1,y1) has at most H cells and at least L many cells for each ingredient
        '''
        x0, x1, y0, y1 = self.order_cords(x0, x1, y0, y1)

        t, m = self.number_of_ingredients(x0, x1, y0, y1)
        if t + m > self.H:
            return False
        if t < self.L or m < self.L:
            return False
        return True

    def slice_pizza(self):
        '''
        Slices the pizza.

        :return: List of slices, each slice is represented as a tuple (x0 y0 x1 y1) where x0<x1 and y0<y1
        '''
        stack = []
        slice_list = []
        stack.append((0, self.height - 1, 0, self.width - 1))

        while stack:
            sl = stack.pop()
            x0, x1, y0, y1 = sl
            t, m = self.number_of_ingredients(*sl)
            if t + m <= self.H and t >= self.L and m >= self.L:  # conditions for a slice are fulfilled
                slice_list.append((x0, y0, x1, y1))
                self.mark_as_covered(x0, x1, y0, y1)
            elif t + m > self.H and not self.has_feasible_ingred(*sl):
                raise Exception('Something went terribly wrong...')
            else:
                direction, pos = self.find_optimal_cut(*sl)
                if direction == 'n':  # no feasible cut exists
                    new_slice = self.place_greedy(*sl)
                    if new_slice:
                        slice_list.append((new_slice[0], new_slice[2], new_slice[1], new_slice[3]))
                        self.mark_as_covered(*new_slice)
                elif direction == 'v':  # vertical cut
                    stack.append((x0, pos, y0, y1))
                    stack.append((pos + 1, x1, y0, y1))
                else:  # horizontal cut
                    stack.append((x0, x1, y0, pos))
                    stack.append((x0, x1, pos + 1, y1))

        return self.optimize_slices(slice_list)

    def place_greedy(self, x0, x1, y0, y1):
        '''
        Computes a biggest feasible slice in a given slice of the pizza. If more than one optimal slice exists, the one
        that has the smallest edge lengths will be chosen.
        :return: optimal slice in (x0,y0)--(x1,y1)
        '''
        size = 0
        max_slices = []
        if not self.has_feasible_ingred(x0, x1, y0, y1):
            raise Exception('Oh shit that did not go as planned...')
        for pos1_x in range(x0, x1 + 1):
            for pos1_y in range(y0, y1 + 1):
                for pos2_x in range(pos1_x, x1 + 1):
                    for pos2_y in range(pos1_y, y1 + 1):
                        if self.is_slice(pos1_x, pos2_x, pos1_y, pos2_y) and (pos2_x - pos1_x + 1) * (
                                pos2_y - pos1_y + 1) > size:
                            size = (pos2_x - pos1_x + 1) * (pos2_y - pos1_y + 1)
                            max_slices = [(pos1_x, pos2_x, pos1_y, pos2_y)]
                        elif self.is_slice(pos1_x, pos2_x, pos1_y, pos2_y) and (pos2_x - pos1_x + 1) * (
                                pos2_y - pos1_y + 1) == size:
                            max_slices.append((pos1_x, pos2_x, pos1_y, pos2_y))

        return min(max_slices, key=lambda t: max(t[2] - t[0], t[3] - t[1])) if max_slices else None

    def mark_as_covered(self, x0, x1, y0, y1):
        '''
        Marks all cells in a given slice as covered. An Exception is raised if the given slice overlaps with already
        covered cells.
        '''

        x0, x1, y0, y1 = self.order_cords(x0, x1, y0, y1)

        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                if self.cover_matrix[i][j] == 1:
                    raise Exception('Sry mate, but the position {} is already covered by some slice...'.format((i, j)))
                self.cover_matrix[i][j] = 1

    def optimize_slices(self, L):
        '''
        Optimizes given set of slices by checking whether the slices can be expanded vertically or horizontally.

        :param L: list of slices, each slice is a tuple (x0, y0, x1, y1)
        :return: list of optimized slices
        '''
        opt_L = []
        for x0, y0, x1, y1 in L:
            upper, left, right, lower = x0, y0, y1, x1
            max_add_cells = self.H - (x1 - x0 + 1) * (y1 - y0 + 1)
            while upper > 0 and not any(self.cover_matrix[upper - 1, y0:y1 + 1]) \
                    and (y1 - y0 + 1) * (x0 - (upper - 1) + lower - x1) <= max_add_cells:
                upper -= 1
            while left > 0 and not any(self.cover_matrix[x0:x1 + 1, left - 1]) \
                    and (x1 - x0 + 1) * (y0 - (left - 1) + right - y1) <= max_add_cells:
                left -= 1
            while right < self.width - 1 and not any(self.cover_matrix[x0:x1 + 1, right + 1]) \
                    and (x1 - x0 + 1) * (y0 - left + (right + 1) - y1) <= max_add_cells:
                right += 1
            while lower < self.height - 1 and not any(self.cover_matrix[lower + 1, y0:y1 + 1]) \
                    and (y1 - y0 + 1) * (x0 - upper + (lower + 1) - x1) <= max_add_cells:
                lower += 1

            exp_vert_val = (y1 - y0 + 1) * (x0 - upper + lower - x1)  # additional cells covered by vertical expansion
            exp_hor_val = (x1 - x0 + 1) * (y0 - left + right - y1)  # additional cells covered by horizontal expansion

            if 0 < exp_vert_val <= exp_hor_val:
                if left - y0: self.mark_as_covered(x0, x1, left, y0 - 1)
                if right - y1: self.mark_as_covered(x0, x1, y1 + 1, right)
                opt_L.append((x0, left, x1, right))
            elif 0 < exp_hor_val < exp_vert_val:
                if upper - x0: self.mark_as_covered(upper, x0 - 1, y0, y1)
                if lower - x1: self.mark_as_covered(x1 + 1, lower, y0, y1)
                opt_L.append((upper, y0, lower, y1))
            else:
                opt_L.append((x0, y0, x1, y1))

        return opt_L
