import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# create 2d array for time studied
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape((-1, 1))
# create 2d array for scores
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape((-1, 1))

#create model and fir time studied and scores


time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.2)
model = LinearRegression()
model.fit(time_train, score_train)
model.score(time_test, score_test)
print(model.score(time_test, score_test))
plt.scatter(time_studied, scores)
plt.plot(np.linspace(0, 70, 100).reshape(-1,1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), color='red')
plt.ylim(0, 100)
plt.show()