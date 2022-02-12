import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axisartist import SubplotZero

# ---------------------Data
training_data = [
    (np.array([1, 4]), 0),
    (np.array([1, 5]), 0),
    (np.array([2, 4]), 0),
    (np.array([2, 5]), 0),
    (np.array([3, 1]), 1),
    (np.array([3, 2]), 1),
    (np.array([4, 1]), 1),
    (np.array([4, 2]), 1),
]

# ---------------------Training

class learning_rule():
    def __init__(self,
                 trainingdata,
                 epochs
                 ):
        self.trainingdata = trainingdata
        self.epochs = epochs

    def initialize(self):
        weight_vector = np.array(np.random.rand(1,2))
        bias = 0.5
        return weight_vector, bias
    def _hardlim(self, x):
        if x <0:
            return 0
        else:
            return 1

    def train_perceptron(self):
        w, b = self.initialize()
        print(f'initial_weights = {w}')
        print(f'initial bias = {b}')
        print(f'training data = {self.trainingdata}')
        errors = {}
        training_process = []
        print('calculating.....')
        for epoch in range(self.epochs):
            for p in range(len(self.trainingdata)):
                n = np.dot(w,self.trainingdata[p][0] + b)
                a = self._hardlim(n)
                e = self.trainingdata[p][1] - a
                w += np.multiply(e,self.trainingdata[p][0])
                b += e
                training_process.append(e)

            if np.all((np.array(training_process) == 0)):
                break
            errors['epoch_'.join(str(epoch))] = training_process

        print(f'Results: weights = {w}   bias = {b}')
        print(errors[str(self.epochs - 1)])
        return w,b
    def test(self, vector,weights,bias):
        n_test = np.dot(weights,vector) + bias
        a_test = self._hardlim(n_test)
        print(f'class ={a_test}')
        return a_test

# ----------------------Tests
training = learning_rule(training_data, 10)
new_weights, new_bias = training.train_perceptron()
test_class = training.test(np.array([2, 5]), new_weights, new_bias)
print(test_class)
#------------------------Plot
xmin, xmax, ymin, ymax = -5,5,-5,5
ticks_frequency = 1
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('$P1$', size=14, labelpad=-24, x=1.02)
ax.set_ylabel('$P2$', size=14, labelpad=-21, y=1.02, rotation=0)

plt.text(0.49, 0.49, r"$O$", ha='right', va='top',
         transform=ax.transAxes,
         horizontalalignment='center', fontsize=14)
x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
ax.set_xticks(x_ticks[x_ticks != 0])
ax.set_yticks(y_ticks[y_ticks != 0])
ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
ax.set_yticks(np.arange(ymin, ymax+1), minor=True)
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

#Points:
p1 = []
for i in range(len(training_data)):
    x = training_data[i][0][0]
    p1.append(x)
p2 = []
for y in range(len(training_data)):
    p2.append(training_data[y][0][-1])
classification = []


print(p1)
print(p2)
def decision_boundary(x, weight_vector, bias):
    y = -(weight_vector[0][0]/weight_vector[0][-1])*x - (bias/weight_vector[0][-1])
    return y
DB_y = [round(decision_boundary(y, new_weights, new_bias),1) for y in x_ticks]

plt.scatter(p1, p2)
plt.plot(x_ticks, DB_y)
plt.show()

