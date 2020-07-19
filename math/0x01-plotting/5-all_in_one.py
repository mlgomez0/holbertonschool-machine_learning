#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(3,2)
ax1.plot(y0, 'r')
ax1.xlim([0,10])
ax2.scatter(x1, y1, s=10, c='m')
ax2.title("Men's Height vs Weight")
ax2.xlabel("Height (in)")
ax2.ylabel("Weight (lbs)")
ax3.plot(x2, y2)
ax3.yscale("log")
ax3.xlim([0, 28650])
ax3.title("Exponential Decay of C-14")
ax3.ylabel("Fraction Remaining")
ax3.xlabel("Time (years)")
ax4.plot(x3, y31, color='r', linestyle='dashed', label='C-14')
ax4.plot(x3, y32, color='g', label='Ra-226')
ax4.legend()
ax4.title("Exponential Decay of Radioactive Elements")
ax4.xlabel("Time (years)")
ax4.ylabel("Fraction Remaining")
ax4.xlim([0,20000])
ax4.ylim([0,1])
ax5.hist(student_grades, range=(0, 100), edgecolor='black')
ax5.xlim([0, 100])
ax5.ylim([0, 30])
ax5.title("Project A")
ax5.ylabel("Number of Students")
ax5.xlabel("Grades")
plt.show()
