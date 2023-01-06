# from sklearn.model_selection import train_test_split
# import numpy as np

# x = np.arange(1, 25).reshape(12, 2)
# y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# print(x)
# print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4, random_state=None,shuffle=True)

# print("x_train :", x_train)
# print("y_train :", y_train)
# print("x_test :", x_test)
# print("y_test :", y_test)
import os
save_folder = 'saved/test_txt '

save_folder = os.path.join(os.path.dirname(__file__), save_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

f = open(os.path.join(save_folder, 'note.txt'), "x")
a = 0.9924
b = 0.9924058462213944
c = 0.9925134064069584
line = [a, b, c]
lines = ["a","b","c"]
with open(os.path.join(save_folder, 'note.txt'), 'a') as f:
    # f.write(('%g ' * len(line)).rstrip() % line + '\n')
    for word  in line:
        index = line.index(word)
        f.write(lines[index] + " = " +str(word))
        f.write("\n")
f.close()