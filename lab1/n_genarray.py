import numpy as np

# Generate a random array with normal
rand_array = np.random.normal(0,1,(3,4))
print("Normal Array:\n",rand_array)

# Generate a random array with uniform
rand_array = np.random.uniform(1,7,3)
print("Uniform Array:\n",rand_array)

# Generate a random array with randint
rand_array = np.random.randn(5,2) + np.array([0, 0])
print("Random Array:\n",np.random.randn(5,2))
print("Random Array expected to be close to 0:\n",rand_array)