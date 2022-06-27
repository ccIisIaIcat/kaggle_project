import numpy as np
import matplotlib.pyplot as plt

original_list = np.array([0.2,0.6,0.2])

convert_matrix = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.1,0.8]])

print(np.matmul(original_list,convert_matrix))

tool_set = []
a1_set = []
a2_set = []
a3_set = []
a4_set = []
a5_set = []
a6_set = []
a7_set = []
a8_set = []
a9_set = []
for i in range(10):
    tool_set.append(i)
    convert_matrix = np.matmul(convert_matrix,convert_matrix)
    a1_set.append(convert_matrix[0][0])
    a2_set.append(convert_matrix[0][1])
    a3_set.append(convert_matrix[0][2])
    a4_set.append(convert_matrix[1][0])
    a5_set.append(convert_matrix[1][1])
    a6_set.append(convert_matrix[1][2])
    a7_set.append(convert_matrix[2][0])
    a8_set.append(convert_matrix[2][1])
    a9_set.append(convert_matrix[2][2])
    print(convert_matrix)

plt.plot(tool_set,a1_set)
plt.plot(tool_set,a2_set)
plt.plot(tool_set,a3_set)
plt.plot(tool_set,a4_set)
plt.plot(tool_set,a5_set)
plt.plot(tool_set,a6_set)
plt.plot(tool_set,a7_set)
plt.plot(tool_set,a8_set)
plt.plot(tool_set,a9_set)
plt.show()