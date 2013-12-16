import matplotlib.pyplot as plt

with open("criterion.txt") as f:
    dataCriterion = f.read()

with open("convergence.txt") as f:
    dataConvergence = f.read()

dataCriterion = dataCriterion.split(',')
dataConvergence = dataConvergence.split(',')
x = list()
for i in range(1,482*20+20) :
    if i%20 == 0 :
        x.append(i)

plt.ylabel('Convergence criterion, logarithmic scale')
plt.xlabel('Iterations')
plt.plot(x, dataConvergence, 'blue')
plt.title("Plot of convergence criterion")
"""
plt.ylabel('Criterion')
plt.xlabel('Iterations')
plt.plot(x, dataCriterion, 'blue')
plt.title("Plot of the criterion")
"""
plt.show()
