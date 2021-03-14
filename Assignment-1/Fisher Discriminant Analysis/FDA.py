import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import copy
import math

file = open("dataset_FLD.csv")
df = np.loadtxt(file, delimiter=",")
row, column = df.shape
class1 = []
class2 = []
ptpos = []
ptneg = []
for i in range(row):
    if df[i][3] == 1:
        class1.append([df[i][0], df[i][1], df[i][2]])
        ptpos.append([df[i][3]])
    else:
        class2.append([df[i][0], df[i][1], df[i][2]])
        ptneg.append([df[i][3]])
class1 = np.array(class1, dtype='float32')
class2 = np.array(class2, dtype='float32')
ptpos = np.array(ptpos, dtype='float32')
ptneg = np.array(ptneg, dtype='float32')

mean1 = np.mean(class1, axis=0)
mean2 = np.mean(class2, axis=0)

diff_means = mean1 - mean2


# calculate Sw
sub1 = class1 - mean1
sub2 = class2 - mean2
Sw = np.dot(sub1.T, sub1) + np.dot(sub2.T, sub2)
Sw = np.linalg.inv(Sw)

# calculate W
W = np.dot(Sw, diff_means)
W = W / np.linalg.norm(W)
print('W')
print(W)

# Plot the 3D points
threeD = plt.figure()
D3 = threeD.add_subplot(111, projection='3d')

D3.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='r', marker='o')
D3.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='b', marker='o')
plt.legend(['Positive', 'Negative'])
plt.show()


projection_1 = np.dot(W, class1.T)
projection_2 = np.dot(W, class2.T)

ax = plt.figure().add_subplot(111)
projection_1.sort()
projection_2.sort()
pdf1 = stats.norm.pdf(projection_1, np.mean(
    projection_1), np.std(projection_1))
pdf2 = stats.norm.pdf(projection_2, np.mean(
    projection_2), np.std(projection_2))


def solve(m1, m2, std1, std2):
    # Calculate roots of 2 normal distributions
    a = 1 / (2 * std1**2) - 1 / (2 * std2**2)
    b = m2 / (std2**2) - m1 / (std1**2)
    c = m1**2 / (2 * std1**2) - m2**2 / (2 * std2**2) + np.log(std1 / std2)
    return np.roots([a, b, c])


# 1D Projections
discpt = solve(np.mean(projection_1), np.mean(projection_2),
               np.std(projection_1), np.std(projection_2))
print('Threshold')
print(discpt[1])
pt = discpt[1]
ax.scatter(projection_1, len(projection_1) * [0], c='r', marker='o')
ax.scatter(projection_2, len(projection_2) * [0], c='b', marker='o')
ax.scatter(pt, 0, c='g', marker='v')
plt.legend(['Positive', 'Negative', 'Threshold'])
plt.title('1D Projections. Threshold = ' + str(pt))
plt.show()

ax = plt.figure().add_subplot(111)
ax.plot(projection_1, pdf1, c='r')
ax.plot(projection_2, pdf2, c='b')
ax.scatter(pt, 0, c='y', marker='x')
plt.axvline(pt, color='g', label='axvline - full height')
plt.legend(['Positive', 'Negative', 'Discriminating Line', 'Threshold'])
plt.axhline(0, color='k', label='axhline - full width')
plt.title('Normal distributions. Threshold = ' + str(pt))
plt.show()


# Discriminating Plane, W and Threshold
alphax = 0
alphay = 0


alphaz = ((alphax * W[0]) + (alphay * W[1])) * (-1) / W[2]
coord = np.linspace(-5, 5, 200)
xline = coord * W[0] + alphax
yline = coord * W[1] + alphay
zline = coord * W[2] + alphaz

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xt = np.linspace(-4, 4, 200)
yt = np.linspace(-4, 4, 200)

X = []
Y = []

X, Y = np.meshgrid(xt, yt)
Z = (- W[0] * X - W[1] * Y) / W[2]


ax.plot_surface(X, Y, Z, alpha=0.25)
ax.plot3D(xline, yline, zline, 'red')
ax.plot(alphax, alphay, alphaz, c='green', marker='x')
plt.legend(['W', 'Threshold'])
plt.title('Discriminating Plane, W and Threshold')

plt.show()

# 3D plot with Discriminating Plane, W and Threshold
fig = plt.figure()
D3 = fig.add_subplot(111, projection='3d')

D3.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='r', marker='o')
D3.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='b', marker='o')
X = []
Y = []
xt = np.linspace(-10, 10, 200)
yt = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xt, yt)
Z = (- W[0] * X - W[1] * Y) / W[2]
D3.plot_surface(X, Y, Z, alpha=0.25)
D3.plot(xline, yline, zline, 'green', linewidth=3.0)
D3.plot(alphax, alphay, alphaz, 'black', marker='x')
plt.legend(['W', 'Threshold', 'Positive', 'Negative', ])
plt.show()


# calculating accuracy
row, col = df.shape
sl = np.array(df[:row, :3])
pred = np.dot(W, sl.T)
array = (pred > discpt[1]).astype(int)

count = 0
for i in range(row):
    if((array[i] == 1) & (float(df[i][3]) == 1)):
        count += 1
    elif((array[i] == 0) & (float(df[i][3]) == 0)):
        count += 1
accuracy = (count * 100 / row)
print('Accuracy')
print(accuracy)
