import numpy as np
np.random.seed(0)

list = np.array([[1, 9], [2, 8], [3, 7], [4, 6], [5, 5], [6, 4], [7, 3], [8, 2], [9, 1], [10, 0]])

np.random.shuffle(list)
print(list)