'''
Question: Magnetic field B = 0.5 Tesla (x=0) z direction. With a linear gradient in the x direction, 
        changing by 3% over a distance of 0.8 meters. 
        A proton enters this field perpendicular to the magnetic field lines with a speed of 2*10**7 m/s
        (a) Calculate the expected Larmor radius and cyclotron period.
        (b) Calculate the expected gradient drift speed of the proton using formulae from lectures.
        (c) Write a small computer program which traces the trajectory of this particle over a number of orbits.
        A procedure for doing this might be as follows:
            - First, choose a time-step for your program, which we will call delta t. A suitable starting value 
                might be 10**(-10) secs but it might not be your final value. 
            - Then use the initial velocity and magnetic field (both vectors) to determine the forcce, and hence the acceleration,
                at time t = 0. Use this acceleration and initial velocity to deduce the new position of the proton at time t = delta t.
                (Note: be careful about v at t = 0 or t = delta t and etc.)
        (d) Show a graph of the drift over (i) 10 cycles and (ii) 100 cycles (use any graphics program
            you like).
        (e) Calculate the mean drift speed and direction over an integral number of cycles and compare
            it to your answer in (b).
        (f) Re-run the program using different magnetic field gradients (in the x direction only) and plot
            a graph of your measured mean drift compared to the predictions given by the formula which
            you used in (b), as a function of the magnetic field gradient. How large does the gradient have
            to be before you start to see significant departures between your model and the theoretical
            formulation? Describe the departures from classical theory which you see. Also indicate what
            you mean by “significant”.
'''
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    q =  1.6*10**(-19)                  # proton charge
    m = 1.672*10**(-27)                 # proton mass
    B = 0.5                             # magnetic field strength
    grad = 0.03                         # gradient coefficient (per 0.8 meters)
    vx = 2*10**7                        # initial velocity (in x direction)

    # (c) Use the RK-4 integration method to calculate the cyclon trajectory of a particle --> cyclotron
    # (d) Show the drift
    #       (i) over 10 cycles (approximatly)

    freq = q*B/m/(2*np.pi)              # calculate the cyclotron frequency (converted to Hz) at the initial position
    cycles = 10
    tau = 10**(-10)
    nSteps = int(cycles/freq/tau)       # estimate the rough number of steps required to reach n cycles
    
    start = time.time()
    trajectory = cyclotron([0,0,0], [vx, 0, 0], q, m, B, nSteps, tau, grad, maxCycle=cycles)            
    print('Computation time - 10 cycles: '+str(time.time() - start))
    print(trajectory.shape)
    
    plt.plot(trajectory[0,:], trajectory[1,:])
    plt.title('Particle Trajectory (10 cycles)')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    #       (ii) over 100 cycles (approximatly)
    cycles = 100
    nSteps = int(cycles/freq/tau)
    start = time.time()
    trajectory = cyclotron([0,0,0], [vx, 0, 0], q, m, B, nSteps, tau, grad, maxCycle=cycles)         
    print('Computation time - 100 cycles: '+str(time.time() - start))
    print(trajectory.shape)

    plt.plot(trajectory[0,:], trajectory[1,:])
    plt.title('Particle Trajectory (100 cycles)')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    # some statistics
    # the computational complexity is linear
    # using tau = 10**(-9), 10 cycles take 0.03 second; 100 cycles take 0.3 seconds (single thread; AMD Ryzen 7 3700X at 4.05 GHz)
    # using tau = 10**(-10), 10 cycles take 0.3 second; 100 cycles take 3 seconds (single thread; AMD Ryzen 7 3700X at 4.05 GHz)
    # using tau = 10**(-11), 10 cycles take 3 seconds; 100 cycles take 30 seconds (single thread; AMD Ryzen 7 3700X at 4.05 GHz)

    # additional numerical accuracy tests
    dt = [10**(-8), 10**(-9), 10**(-10), 10**(-11)]
    cycles = 10
    for tau in dt:
        nSteps = int(cycles/freq/tau)
        trajectory = cyclotron([0,0,0], [vx, 0, 0], q, m, B, nSteps, tau, grad, maxCycle=cycles) 
        plt.plot(trajectory[0,:], trajectory[1,:], label = 'dt = '+str(tau))

    plt.title('Particle Trajectory Numerical Accuracy (10 cycles)')
    plt.legend()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    # (e) calculate the mean drift speed 
    # 100 cycles
    vol_drift = num_drift_vol(trajectory, tau, axis=1)                  # average drift velocity over 100 cycles
    anal_vol_drift = anal_drift_vol(vx, q, m, B, grad)                  # theoretical/analytical drift velocity
    print('(d): the average estimated drift velocity from numerical results is: '+str(vol_drift)+' m/s')
    print('     The velocity calculated in (b) is approximatly ' +str(anal_vol_drift)+' m/s, and the difference is '+str(vol_drift-anal_vol_drift)+' m/s')

    # (f) re-run for different magnetic field gradients and show the discrepancies between the theoretical and numerical results
    freq = q*B/m/(2*np.pi)              # calculate the cyclotron frequency (converted to Hz) at the initial position
    cycles = 10
    tau = 10**(-10)
    nSteps = int(cycles/freq/tau)

    fig, ax = plt.subplots(2)
    fig.set_size_inches(14, 8)
    for i in range(8):
        grad = 0.01*i           # 1% increment
        trajectory = cyclotron([0,0,0], [vx, 0, 0], q, m, B, nSteps, tau, grad, maxCycle=cycles)
        vol_drift = num_drift_vol(trajectory, tau, axis=1)
        anal_vol_drift = anal_drift_vol(vx, q, m, B, grad)
        ax[0].plot(trajectory[0,:], trajectory[1,:], label=str(grad))
        ax[1].scatter(i, abs(vol_drift-anal_vol_drift), label='Gradient coeff: '+str(-grad))              # percentage of decrease per 0.8 m
    ax[0].set_ylabel('Y axis')
    ax[0].set_xlabel('X axis')
    ax[1].set_ylabel('Drift Velocity Difference')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Incident Number')
    ax[0].set_title('Numerical Particle Trajectory and Drift Velocity Discrepancies')
    plt.legend()
    plt.show()

    return 

# Integration method
def rk4(x, t, tau, derivsRK, args=None):
    '''Runge-Kutta Integrator (4th order)
    Input: 
                        x                               state vector [position x, pos y, pos z, velocity u, vel v, vel w]
                        t                               time of state
                        tau                             time step size
                        derivsRK                        the ODE model to be integrated (the 'RHS' of an ODE)
                        args            (optional)      additional arguments to pass on to the ODE model
    Return:
                        xout                            updated state vector [position x, pos y, pos z, velocity u, vel v, vel w]
    '''
    
    half_tau = 0.5*tau
    F1 = derivsRK(x,t, args[0],args[1],args[2], args[3])            # first order
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half, args[0],args[1],args[2], args[3] )   # second order
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half, args[0],args[1],args[2], args[3] )   # third order
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full, args[0],args[1],args[2], args[3] )   # fourth order
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))

    return xout

# particle trajectory
def particle_acc(state, t, q, m, B, grad):
    '''Particle acceleration model.
    Input:
                        state                           state vector [position x, pos y, pos z, velocity u, vel v, vel w]
                                                        pos             the position in 3D [x, y, z]
                                                        vel             the velocity in 3D  [u, v, w]
                        q                               particle charge
                        m                               particle mass
                        B                               magnetic field strength
                        grad                            the magnetic field gradient coeff in the x direction
    Return:
                        stateOut                        updated state vector [position x, pos y, pos z, velocity u, vel v, vel w]'''
    B = B*(1-grad*state[0]/0.8)                         # the B magnitude at position pos
    accoef = q/m*B
    acc = [-accoef*state[4], accoef*state[3], 0]
    stateOut = np.array([state[3], state[4], state[5], acc[0], acc[1], acc[2]])

    return stateOut

def cyclotron(pos,v, q, m, B, nStep, tau, grad, maxCycle=None):
    '''The cyclotron trajectory of the particle.
    Input:
                        v                               the initial velocity
                        pos                             the initial position
                        q                               paricle charges
                        m                               mass of the particle
                        B                               the magnetic field strength
                        nStep                           the total number of steps 
                        tau                             time step size
                        grad                            the magnetic field gradient coeff in the x direction (per 0.8 m)
                        maxCycle       (optional)       the total number of cycles to calculate
    Return:         
                        traj                            the trajectory of the particle
    '''
    state = np.array([pos[0], pos[1], pos[2], v[0], v[1], v[2]])
    # values avaliable: m, B, dt
    t = 0
    traj = np.empty((3, nStep))

    sign = np.sign(state[0])                # sign for cycle counting
    if sign == 0:
        sign+=1
    period = 0                              # number of periods for cycle counting

    for i in range(nStep):                  # iterate particle positions
        state = rk4(state, t, tau, particle_acc,args=[q, m, B, grad])
        traj[0, i] = state[0]
        traj[1, i] = state[1]
        traj[2, i] = state[2]

        if maxCycle is not None:                # detect x axis sign changes - cycle counting
            new_sign = np.sign(state[0])
            if new_sign == 0:
                new_sign+=1

            if new_sign != sign:
                period +=1
                sign = new_sign
            if period/2 == maxCycle:            # end early if desired number of cycles is reached
                print('Has reached '+str(maxCycle)+' cycles.')
                return traj[:, :i]              # cut off empty sections

    return traj

def num_drift_vol(array, tau, axis=1):
    '''The average drift velocity from numerical results.
    Input:
                        array                               numerical trajectory array
                        tau                                 time step size 
                        axis            (optional)          axis number (1 = y)
    Return:
                        vol_drift                           the average drift velocity 
    '''
    dis = array[axis, -1] - array[axis, 0]              # the total drift distance
    vol_drift = dis/len(array[1])/tau                   # average drift velocity over n cycles

    return vol_drift

def anal_drift_vol(vol, q, m, B, grad):
    '''The analytical drift velocity.
    Input:
                        vol                                 initial velocity (x direction)
                        q                                   paricle charges
                        m                                   mass of the particle
                        B                                   the magnetic field strength
                        grad                                the magnetic field gradient coeff in the x direction (per 0.8 m)
    Return:
                        vol_drift                           the theoretical drift velocity
    '''
    vol_drift = 0.5*(((m*vol)/(q*B))**2*(q*B/m))*(B*grad/0.8)/B

    return vol_drift

if __name__ == '__main__':
    main()