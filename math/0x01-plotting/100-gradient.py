#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(x, y, s=40, c=z, cmap=cm)
clb = plt.colorbar(sc)
plt.title("Mountain Elevation")
plt.xlabel("x coordinate (m)")
plt.ylabel("y coordinate (m)")
clb.set_label('elevation (m)')
plt.show()
