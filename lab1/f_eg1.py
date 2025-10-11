import numpy as np

# Creating a 1D array from a list
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", array_1d)

# Creating a 2D array (list of lists)
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:\n", array_2d)

# Creating a 1D array of zeros
zeros_1d = np.zeros(5)
print("1D Array of zeros:", zeros_1d)

# Creating a 2D array of zeros
zeros_2d = np.zeros((3, 4))  # 3 rows, 4 columns
print("2D Array of zeros:\n", zeros_2d)

# Creating a 2D array of ones
ones_2d = np.ones((2, 3))  # 2 rows, 3 columns (a 2 by 3 matrix)
print("2D Array of ones:\n", ones_2d)

# Creating a 3D array of ones
ones_2d = np.ones((4, 2, 3))  # a 4 by 2 by 3 tensor
print("3D Array of ones:\n", ones_2d)

print(ones_2d.shape)
