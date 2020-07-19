#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

a = fruit[0]
b = fruit[1]
o = fruit[2]
p = fruit[3]
ind = ["Farrah", "Fred", "Felicia"]
p1 = plt.bar(ind, a, 0.5, color='r')
p2 = plt.bar(ind, b, 0.5, color='#FFFF00', bottom=a)
p3 = plt.bar(ind, o, 0.5, color='#ff8000', bottom=[
    a[j] + b[j] for j in range(len(a))])
p4 = plt.bar(ind, p, 0.5, color='#ffe5b4', bottom=[
    a[j] + b[j] + o[j] for j in range(len(a))])
plt.yticks(np.arange(0, 81, 10))
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.legend((p1[0], p2[0], p3[0], p4[0]), (
    'apples', 'bananas', 'oranges', 'peaches'))
plt.show()
