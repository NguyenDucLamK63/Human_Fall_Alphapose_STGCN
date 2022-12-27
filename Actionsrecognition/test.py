from sklearn.model_selection import train_test_split
import numpy as np

x = np.arange(1, 25).reshape(12, 2)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4, random_state=None,shuffle=True)

print("x_train :", x_train)
print("y_train :", y_train)
print("x_test :", x_test)
print("y_test :", y_test)
