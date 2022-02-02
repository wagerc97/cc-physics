""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
pendulum.py
sheet 1 exercise 1  'Pendulum'
CO-Physics 1 2021WS
By Clemens Wager, 01635477
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt

# Set up configuration options and special features
# standard values
NumericalMethod = 3
theta0 = 20  # angle in degree
tau = 0.004  # time step in seconds
nstep = 5000  # number of time steps
Td = 0.2        # driving force period
A0 = 100*9.81   # amplitude of driving force

plotExtraLines = False  # Plot the zero line, the positive and negative theta0
plotOmega = False       # Make a second plot of angular displacement (omega) over time

print(f"=== Standard parameters ===\n" +
      f"Method: Euler-Cromer\n" +
      f"Initial angle: {theta0} degrees\n" +
      f"Period of driving force: {round(Td,4)}\n" +
      f"Amplitude of driving force: {A0/9.81}*g\n" +
      f"Time step: {tau} s\n" +
      f"Number of time steps: {nstep}\n")

answer = input("Use standard parameters? [Y]es or [N]o? ").lower()
if answer in ('yes', 'y', ''):
    print("Starting simulation with standard parameters...\n")

# Select the numerical method to use: Euler or Verlet
if answer not in ('yes', 'y', ''):
    NumericalMethod = input('Choose a numerical method (1: Euler; 2: Verlet; 3: Euler-Cromer): ')
    NumericalMethod = int(NumericalMethod)

# * Set initial position and velocity of pendulum
if answer not in ('yes', 'y', ''):
    theta0 = input('Enter initial angle (in degrees): ')
    theta0 = float(theta0)
theta = theta0 * np.pi / 180  # Convert angle to radians
omega = 0.0  # Set the initial velocity

# * Set the physical constants and other variables
g = 9.81
L = 9.81
g_over_L = 1.0  # The constant g/L
A_over_L = A0 / L
time = 0.0  # Initial time
irev = 0  # Used to count number of reversals
if answer not in ('yes', 'y', ''):
    Td = input('Enter period of driving force: ') # period of driving force
    Td = float(Td)
    A0 = input('Enter amplitude of driving force (g): ')  # amplitude of driving force
    A0 = float(A0)*g
    tau = input('Enter time step: ') # time step
    tau = float(tau)

# Calculate acceleration w.r.t. gravity and the driving pivot
accel = -(g_over_L + A0 * np.sin(2 * np.pi * time / Td)) * np.sin(theta)
# * Take one backward step to start Verlet
theta_old = theta - omega * tau + 0.5 * accel * tau ** 2

# * Loop over desired number of steps with given time step
#    and numerical method
if answer not in ('yes', 'y', ''):
    nstep = input('Enter number of time steps: ')
    nstep = int(nstep)
    print("Simulating...")
t_plot = np.empty(nstep)    # record time steps
th_plot = np.empty(nstep)   # record theta
om_plot = np.empty(nstep)   # record omega
period = np.empty(nstep)    # record period estimates
reversalTimes = []  # records the timestamps for reversals

# Bools to record reversals
if theta > 0:
    positiveSide = True
else: # theta < 0
    positiveSide = False

# Start propagation of angle theta over time
for istep in range(nstep):

    # * Record angle and time for plotting
    t_plot[istep] = time    # record time
    th_plot[istep] = theta * 180 / np.pi  # Convert angle to degrees and record it
    om_plot[istep] = omega  # record omega
    time = time + tau
    # update acceleration
    accel = -(g_over_L + A0  * np.sin(2 * np.pi * time / Td)) * np.sin(theta)

    # * Compute new position and velocity
    if NumericalMethod == 1:  # EULER
        """ Euler method is not suited for opens systems """
        theta_old = theta  # Save previous angle
        theta = theta + tau * omega  # Euler method
        omega = omega + tau * accel
    elif NumericalMethod == 2:  # VERLET
        """ Verlet only computes positions, not velocity"""
        theta_new = 2 * theta - theta_old + tau ** 2 * accel
        theta_old = theta  # Verlet method
        theta = theta_new
    else:  # if NumericalMethod == 3:  # EULER-CROMER
        """ E-C method computes position and velocity"""
        # update omega
        omega = omega - (g_over_L + accel / L) * np.sin(theta) * tau
        # update theta
        theta = theta + omega * tau
        # update time already at start of function

    # Test if the pendulum has passed through theta = 0
    if positiveSide:
        if theta < 0:  # past zero to negative side
            positiveSide = False
            print('Turning point at time t =', round(time, 4))
            reversalTimes.append(time)
        if irev == 0:  # If this is the first change,
            time_old = time  # just record the time
        else:
            period[irev - 1] = 2 * (time - time_old)
            time_old = time
    else:  # negative Side
        if theta > 0:  # past zero to negative side
            positiveSide = True
            print('Turning point at time t =', round(time, 4))
            reversalTimes.append(time)
        if irev == 0:  # If this is the first change,
            time_old = time  # just record the time
        else:
            period[irev - 1] = 2 * (time - time_old)
            time_old = time

        irev = irev + 1  # Increment the number of reversals

# Estimate period of oscillation, including error bar
revN = len(reversalTimes)  # number of reversals
if revN:
    print("Number of reversals:", revN)
    period_diffs = [reversalTimes[i] - reversalTimes[i - 1] for i in range(1, revN)]
    if len(period_diffs):
        AvePeriod = np.mean(period_diffs) * 2
        ErrorBar = np.std(AvePeriod) / np.sqrt(revN)  # standard error of mean
        print(f"Average period = {round(AvePeriod, 4)} +/- {ErrorBar} s")

# Graph the oscillations as theta versus time
print("\n...check the new plot!")
plt.figure(figsize=(8, 6))
plt.title(r'Angle $\theta$ over time')
if plotExtraLines:
    # plot reference
    X_ref = np.linspace(0, nstep*tau, nstep)
    plusY_ref = np.linspace(theta0, theta0, nstep)
    minusY_ref = np.linspace(-theta0, -theta0, nstep)
    Y_zero = np.zeros(nstep)
    plt.plot(X_ref, plusY_ref,  color='black', label=f'+{theta0}°')
    plt.plot(X_ref, Y_zero,     color='red',   label=f'0°')
    plt.plot(X_ref, minusY_ref, color='black', label=f'-{theta0}°')
# plot data
plt.scatter(t_plot, th_plot, s=2, label=r'Pendulum $\theta$')
#plt.ylim([23,27])
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta$ (degrees)')
plt.grid(which='both')
plt.legend(loc='upper right')
plt.show()

if plotOmega:
    # plot angular displacement over time (omega)
    plt.figure(figsize=(8, 6))
    plt.title(r'Angular velocity $\omega$ over time')
    plt.plot(t_plot, om_plot, 'red', label=r'Pendulum $\omega$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\theta$/s (degrees/s)')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()