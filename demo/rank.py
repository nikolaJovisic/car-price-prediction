import numpy as np

arr = np.zeros((6, 8))
arr[0:3, 0] = 1
arr[3:, 1] = 1
arr[:, 2:] = np.eye(6, 6)
print(arr)
print('rank:', np.linalg.matrix_rank(arr, tol=0.1))
