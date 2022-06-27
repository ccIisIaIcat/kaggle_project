import numpy as np
import matplotlib.pyplot as plt

original_list = np.array([0.2,0.6,0.2])

convert_matrix = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.1,0.8]])

print(np.matmul(original_list,convert_matrix))

tool_set = []
a1_set = []
a2_set = []
a3_set = []
for i in range(50):
    tool_set.append(i)
    original_list = np.matmul(original_list,convert_matrix)
    a1_set.append(original_list[0])
    a2_set.append(original_list[1])
    a3_set.append(original_list[2])
    print(original_list)

plt.plot(tool_set,a1_set)
plt.plot(tool_set,a2_set)
plt.plot(tool_set,a3_set)
plt.show()



