import numpy as np
import matplotlib.pyplot as plt

### question 1 RMSE vs lambda in regularization
x = input().strip()
x_axis = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis = np.array(y.split(' ')).astype(float)

### question 2 for GD with LR 0.05 ii) IRLS

x = input().strip()
x_axis11 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis11 = np.array(y.split(' ')).astype(float)

x = input().strip()
x_axis1 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis1 = np.array(y.split(' ')).astype(float)

### question 3 RMSE vs LR for (i) linear,(ii) quadratic and (iii)cubic combinations of the features
x = input().strip()
x_axis2 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis2 = np.array(y.split(' ')).astype(float)

x = input().strip()
x_axis3 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis3 = np.array(y.split(' ')).astype(float)

x = input().strip()
x_axis4 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis4 = np.array(y.split(' ')).astype(float)

### question 4 RMSE vs LR for (i) mean cubic (ii) mean sabsolute and (iii)mean squared cost functions

x = input().strip()
x_axis5 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis5 = np.array(y.split(' ')).astype(float)

x = input().strip()
x_axis6 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis6 = np.array(y.split(' ')).astype(float)

x = input().strip()
x_axis7 = np.array(x.split(' ')).astype(float)

y = input().strip()
y_axis7 = np.array(y.split(' ')).astype(float)


f, (ax) = plt.subplots(1, sharex=True, sharey=True)
f, (ax11) = plt.subplots(1, sharex=True, sharey=True)
f, (ax1) = plt.subplots(1, sharex=True, sharey=True)
f, (ax2) = plt.subplots(1, sharex=True, sharey=True)
f, (ax3) = plt.subplots(1, sharex=True, sharey=True)
f, (ax4) = plt.subplots(1, sharex=True, sharey=True)
f, (ax5) = plt.subplots(1, sharex=True, sharey=True)
f, (ax6) = plt.subplots(1, sharex=True, sharey=True)
f, (ax7) = plt.subplots(1, sharex=True, sharey=True)
f, (ax8) = plt.subplots(1, sharex=True, sharey=True)
f, (ax9) = plt.subplots(1, sharex=True, sharey=True)

ax.plot(x_axis, y_axis, 'o-')
ax.set_xlabel('lambda')
ax.set_ylabel('RMSE in regularization')

ax11.plot(x_axis11, y_axis11, 'o-')
ax11.set_xlabel('number of iterations')
ax11.set_ylabel('RMSE in IRLS')

ax1.plot(x_axis1, y_axis1, 'o-')
ax1.set_xlabel('number of iterations')
ax1.set_ylabel('RMSE in gradient descent with LR 0.05')

ax2.plot(x_axis2, y_axis2, 'o-')
ax2.set_xlabel('Learning rate')
ax2.set_ylabel('RMSE in linear features')

ax3.plot(x_axis3, y_axis3, 'o-')
ax3.set_xlabel('Learning rate')
ax3.set_ylabel('RMSE in quadratic features')

ax4.plot(x_axis4, y_axis4, 'o-')
ax4.set_xlabel('Learning rate')
ax4.set_ylabel('RMSE in cubic features')

ax8.plot(x_axis2,y_axis2,'r')
ax8.plot(x_axis3,y_axis3,'g')
ax8.plot(x_axis4,y_axis4,'b')
ax8.set_title('features')


ax5.plot(x_axis5, y_axis5, 'o-')
ax5.set_xlabel('Learning rate')
ax5.set_ylabel('RMSE in mean cubic cost function')

ax6.plot(x_axis6, y_axis6, 'o-')
ax6.set_xlabel('Learning rate')
ax6.set_ylabel('RMSE in mean absolute cost function')

ax7.plot(x_axis7, y_axis7, 'o-')
ax7.set_xlabel('Learning rate')
ax7.set_ylabel('RMSE in mean squared cost function')

ax9.plot(x_axis5,y_axis5,'r')
ax9.plot(x_axis6,y_axis6,'g')
ax9.plot(x_axis7,y_axis7,'b')
ax9.set_title('cost function')

plt.show()

