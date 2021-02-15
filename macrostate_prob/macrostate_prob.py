import numpy as np
import matplotlib.pyplot as plt

# Statement of question:
# Two interacting two-state paramagnet, each contains 100 elementary magnetic dipoles,
# Total number of units of energy relative to the state with all dipoles pointing up is 80.
# Plot q1 (unit of energy of the first magnet) vs. total macro states. What is the most 
# probable macro state?

m1 = np.empty(81); m2 = np.empty(81)

for q1 in range(81):
    q2 = abs(q1 - 80)
    m1[q1] = np.math.factorial(100)/(np.math.factorial(q1)*np.math.factorial(100 - q1))     #multiplicity function
    m2[q1] = np.math.factorial(100)/(np.math.factorial(q2)*np.math.factorial(100 - q2))     #multiplicity function

totalM = m1*m2
# find the most probable state
print('The most probable macro state is q1 = q2 at position: ' + str(np.where(totalM == totalM.max())))
# find the least probable state
print('The least probable macro state is at position: ' + str(np.where(totalM == totalM.min())))

# plot
fig, ax = plt.subplots()
ax.plot(range(81), totalM)
ax.set_ylabel('Total Macro State')
ax.set_xlabel('qA')     # = q1
plt.show()