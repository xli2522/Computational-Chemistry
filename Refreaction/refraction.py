from cProfile import label
import numpy as np
import matplotlib.pylab as plt

thetaI = np.linspace(0, np.pi/2, 100)               # incident angle
beta = 1.33
alpha = np.sqrt(1-(1/1.33*(np.sin(thetaI)))**2)/np.cos(thetaI)

R = ((alpha - beta)/(alpha + beta))**2
T = alpha*beta*(2/(alpha+beta))**2

def brewster(beta, n1, n2):
    angle = np.arcsin(np.sqrt((1-beta**2)/((n1/n2)**2-beta**2)))
    
    return angle

brewster_angle = brewster(beta, 1, 1.33)
print(brewster_angle)

angle_pos = brewster_angle*2/np.pi*100
plt.plot(R, label = 'R')
plt.plot(T, label = 'T')
plt.plot(T+R, label = 'T + R')
plt.scatter(angle_pos, 0, label = 'Brewster')
plt.xlabel(r'Angle pi/2/100 ')
plt.ylabel('Coefficient Number')
plt.title('Reflection and Transmission Coefficients from Air to Water')
plt.legend()
plt.show()

