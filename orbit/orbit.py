'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
orbit_final.py
sheet 1 exercise 2  "Satellite motion"
CO-Physics 1 2021WS
By Clemens Wager, 01635477
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Objective:
# Modify the program orbit.f or write a new program so that instead of running for
#   a fixed number of time steps, the program stops when the satellite completes one full orbit.
#
# a) The program should compute the following orbital measures:
#   - period
#   - eccentricity
#   - semimajor (+semiminor) axis
#   - perihelion distance
#   Use the Euler-Cromer method and test the program with circular and slightly elliptical orbits.
#   Compare the measured eccentricity with the given value e.
#
# b) Show that your code confirms Kepler's third law.
#   "The square of a planet's orbital period is proportional to
#    the cube of the length of the semi-major axis of its orbit."
#
# c) [optional] Confirm the given equation of the "viral theorem".

import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt

### Global constants ###
G = 6.67408e-11  # gravitational constant in m^3/kg*s^2
M = 1.99e30  # mass of sun in kg
GM = 4 * pi ** 2  # constant (AU^3/yr^2)

### Create satellite ###
m = 1.0  # mass of the satellite in kg
t_0 = 0.0  # 0  # initial time in astronomical units [years]
r_0 = 1  # 1 initial position in astronomical units [AU]
v_0 = 2*pi  # 2*pi initial velocity in [AU/years]

### Lists ###
rplot2 = []  # list with r values
thplot2 = []  # list with theta values
timeplot2 = []  # list with time values (tplot)
kinetic2 = []  # list of kinetic energy
potential2 = []  # list of potential energy
totalenergy = [] # list of total energy

#==================== Simulation parameters ===================#
# General settings
givenData = True    # apply given test data of assignment
interestingSimulations = False  # apply interesting orbit data

dt = 0.005  # 0.005     # size of timestep, adjusts accuracy
orbits = 1              # number of orbits per run
runs = 1  # 4-5         # number of trajectories computed
overlap = False         # plot simulations of each run as if they had started at t_0 simultaneously

# Settings for b)
confirmKeplersThirdLaw = False  # calculate and plot to confirm Kepler's Third Law
kepler_reference = True  # add a line to the Kepler plot that illustrates ideal results
#==============================================================#


# apply given test data of assignment
if givenData:
    """ Resulting values you should obtain:
    period T = 0.58 years  # result=0.59 years
    eccentricity e = 0.436  # result=0.436 """
    r_0 = 1
    v_0 = 3*pi/2
    dt = 0.005

if interestingSimulations:
    # apply interesting simulations inputs for position and velocity
    #                  r_0   v_0
    interestingVars = [[1., 2.0 * pi],  # circular motion
                       [.8, 1.0 * pi],  # planet escapes after ca. 8 orbits
                       [.6, 1.4 * pi],  # "flower"
                       [.6, 1.2 * pi],  # small/big loops alternating with escape
                       [2., 0.4 * pi]]  # "middle bump"
    choice = 3
    orbits = 7
    r_0 = interestingVars[choice][0]
    v_0 = interestingVars[choice][1]

####################### Compute orbit #########################
def euler_cromer(GM, dt, time, r, v, normR):
    """ implement Euler-Cromer method in function """
    # update acceleration
    accel_x = -GM * r[0] / (normR ** 3)    # accel x
    accel_y = -GM * r[1] / (normR ** 3)    # accel y
    # update velocity
    v[0] = v[0] + dt * accel_x      # update v_x
    v[1] = v[1] + dt * accel_y      # update v_y
    # update position
    r[0] = r[0] + dt * v[0]         # update r_x        # The Euler-Cromer step (using updated velocity)
    r[1] = r[1] + dt * v[1]         # update r_y        # The Euler-Cromer step (using updated velocity)
    time = time + dt                # update time
    return time, r, v   # return updated values


def compute_orbit(r_0, v_0):
    """
    Compute the orbit, so the angular displacement in each time step.
    Return:
        r_max: average maximum radius
        r_min: average minimal radius
        period_avg: average period length
    """
    r = [r_0, 0]  # define initial position vector
    v = [0, v_0]  # define initial velocity vector
    normR = math.sqrt(r[0] * r[0] + r[1] * r[1])  # initial position
    normV = math.sqrt(v[0] * v[0] + v[1] * v[1])  # initial energy
    time = t_0  # set time to t_0
    period_agg = 0 # added timesteps calculate period length
    stepmax = 0 # declare variable for max iteration number
    for run in range(runs):  # runs
        # create sublists for this run
        rplot2.append([])
        thplot2.append([])
        timeplot2.append([])
        kinetic2.append([])
        potential2.append([])
        totalenergy.append([])

        if overlap:  # all runs start from t_0
            time = t_0  # set time to t_0
        th_is_negative = False  # for full orbit detection
        porbits = orbits  # set parameter orbits to orbits

        stepmax = 0 # reset max iterations

        # Compute positions on one full orbit
        while porbits:  # loop stops if last angle surpasses 360°
            if stepmax > 3000*orbits: # safeguard
                break
                #raise ValueError(f"Error: Satellite left orbit in run {run}!")
            stepmax += 1
            # Record position, angle, timestep and energy for plotting
            rplot2[run].append(normR)
            thplot2[run].append(math.atan2(r[1], r[0]))  # arctan2 is a jumping function
            timeplot2[run].append(time)
            kinetic2[run].append(0.5 * m * normV ** 2)
            potential2[run].append(-GM * m / normR)
            totalenergy[run].append((0.5 * m * normV ** 2)-(GM * m / normR))
            period_agg += dt  # add timestep to calculate period

            # Test if a full orbit is completed 360 [°] = 2pi [rad]
            # FIRST condition for loop break: orbit past 1Pi
            if thplot2[run][-1] < 0:
                th_is_negative = True  # store that angle surpassed 1Pi
            # SECOND condition: angle above zero again => orbit past origin
            if th_is_negative and thplot2[run][-1] >= 0:
                th_is_negative = False  # set for next orbit
                # subtract one orbit from total number to indicate that one orbit was completed
                # breaks the loop when porbits reaches 0
                porbits -= 1

            # Compute next step
            time, r, v = euler_cromer(GM, dt, time, r, v, normR)

            # Calculate new position and energy from the last recorded incidence
            normR = math.sqrt(r[0] * r[0] + r[1] * r[1])
            normV = math.sqrt(v[0] * v[0] + v[1] * v[1])

    # Aggregate all max. and min. radius averaged over all runs
    r_max_agg = 0
    r_min_agg = 0
    for run in range(runs):
        r_max_agg += max(rplot2[run])
        r_min_agg += min(rplot2[run])
    r_max = r_max_agg / runs # averaged r_max
    r_min = r_min_agg / runs # averaged r_min

    # Calculate average period from lists of timesteps
    period_avg = period_agg / (runs * orbits)  # averaged period
    #print(f"stepmax: {stepmax}\n") #DEBUGGING
    return r_max, r_min, period_avg


########################## Plot data ##########################

# plot function for new lists
def plot_position():
    """ Plot position r from rplot2 over time t from timeplot2. """

    plt.rcParams["figure.figsize"] = (8,8)  # set figure size
    # smart plot loop
    for run in range(runs):
        plt.plot(timeplot2[run],    # time t
                 rplot2[run],       # postion r
                 label="run_%s" % (run + 1))  # set label according to run number

    # plot an origin line
    if r_0 == 1:  # only with standard initial position
        origin_x = [0, period*orbits*runs]
        origin_y = [1, 1]
        plt.plot(origin_x, origin_y,
                 color='black',
                 markersize='2',
                 label='origin')
    # fancy plot
    plt.title('Position over time')
    plt.xlabel("time [yr]")
    plt.ylabel("position [AU]")
    plt.legend(loc='lower right')
    plt.grid()
    #plt.savefig("plot_position_over_time")
    plt.show()


# plot angle NOT IN USE
def plot_angle():
    """ Plot angle theta from thplot2 over time t from timeplot2. """

    plt.rcParams["figure.figsize"] = (8,8)  # set figure size

    # smart plot loop
    for run in range(runs):
        plt.plot(timeplot2[run],    # time t
                 thplot2[run],      # anlge theta
                 label="run_%s" % (run + 1),  # set label according to run number
                 marker='o',
                 color='r')

    # plot an origin line
    if r_0 == 1:  # only with standard initial position
        if overlap:  # origin line short
            origin_x = [0, 0]
            origin_y = [1, 0]
        else:  # origin line long
            origin_x = np.arange(0, (runs + 1))
            origin_y = np.zeros(runs + 1)

        plt.plot(origin_x, origin_y,
                 color='black',
                 markersize='2',
                 label='origin')
    # make plot look nicer
    plt.title('Position as wave over time')
    plt.xlabel("time [yr]")
    plt.ylabel("angle theta")
    plt.legend(loc='lower right')
    plt.show()


# polar plot for data
# more info: http://devdoc.net/python/matplotlib-2.0.0/examples/pylab_examples/polar_demo.html
def plot_polar():
    """ Plot angle theta from thplot2 over time t from timeplot2 on a polar plot. """
    ### setting the axes projection as polar
    # plt.axes(projection='polar')
    plt.rcParams["figure.figsize"] = (8,8)  # set figure size

    ax = plt.subplot(111, projection='polar') # create special polar plot

    # smart plot loop
    for run in range(runs):
        ax.plot(thplot2[run], rplot2[run],
                label=f"run_{run+1}")  # set label according to run number

    ### make plot look nicer
    if confirmKeplersThirdLaw:
        radmax = 3.5
    else: # normal orbit simulation
        radmax = 1.5
    ax.set_rmax(radmax)  # set max radius displayed

    # plt.title('Polar plot of satellite orbit')     # set title
    ax.set_title('Polar plot of satellite orbit')  # set title

    # plt.grid(True) # enable grid
    ax.grid(True)  # enable grid
    if confirmKeplersThirdLaw:
        ax.set_rticks(np.arange(0, radmax, 0.5))  # number of radial ticks
    else: # normal orbit simulation
        ax.set_rticks(np.arange(0, radmax, 0.25))  # number of radial ticks
    ax.set_rlabel_position(+20)  # Move radial labels away from plotted line
    if confirmKeplersThirdLaw:
        ax.set_yticklabels(['.0', '.5', 'Pi', '1.5', '2Pi', '2.5', '3Pi'])  # radial labels
    else: # normal orbit simulation
        ax.set_yticklabels(['.0', '.25', '.5', '.75', 'Pi', '1.25'])  # radial labels

    ax.set_xticks(np.pi / 180 * np.linspace(0, 360, 16, endpoint=False))  # set number of radial lines
    ax.set_xticklabels(
        ['0°', None, '45°', None, '90°', None, '135°', None, '180°', None, '225°', None, '270°', None, '315°',
         None])  # alter radial labels

    if not confirmKeplersThirdLaw:
        plt.legend(loc='upper left')
        #plt.savefig("plot_polar")
        plt.show()


def plot_kepler(periods, semimjraxes):
    """ This plot plot the respective data to confirm Kepler's third law.
    Parameters:
        periods: input list with all periods
        semimjraxes:
    """
    # DEBUGGING
    #print(f"periods:\n{periods}\n")
    #print(f"semimjraxes:\n{semimjraxes}")

    plt.rcParams["figure.figsize"] = (8,8)  # set figure size
    plt.loglog(periods, semimjraxes,
             label='(computed semi-major axis)e(3/2) over computed orbital period',
             marker='o',
             color='g')

    if kepler_reference:
        x = np.linspace(0, periods[-1], len(periods)) # x values
        y = x ** (3 / 2) # y values
        plt.loglog(x, y,
                   label='y*e(3/2) over x',
                   color='black',
                   marker='x')

    # fancy plot
    plt.title('b) Kepler\'s Third Law')
    plt.xlabel("Square of period [log years^2]")
    plt.ylabel("Cube of semimajor axis [log AU^3]")
    plt.legend(loc='upper left')
    plt.grid(which='minor')
    #plt.savefig("plot_kepler_ratio")
    plt.show()


# plot angle NOT IN USE
def plot_energy():
    """ Plot potential, kinetic and total energy over time. """
    plt.rcParams["figure.figsize"] = (8,8)  # set figure size
    # smart plot loop for KE
    for run in range(runs):
        plt.plot(timeplot2[run],    # time t
                 kinetic2[run],     # KE
                 label="KE_run_%s" % (run + 1),  # set label according to run number
                 #marker='.',
                 color='black')
    # smart plot loop for PE
    for run in range(runs):
        plt.plot(timeplot2[run],    # time t
                 potential2[run],   # PE
                 label="PE_run_%s" % (run + 1),  # set label according to run number
                 #marker='.',
                 color='red')
    # smart plot loop for PE
    for run in range(runs):
        plt.plot(timeplot2[run],    # time t
                 totalenergy[run],   # total energy
                 label="Total Energy run_%s" % (run + 1),  # set label according to run number
                 #marker='o',
                 color='blue')
    # fancy plot
    plt.title('Energies over time')
    plt.xlabel("time [yr]")
    plt.ylabel("Energy [M AU^2 / yr^3")
    plt.legend()
    plt.grid(which='both')
    plt.show()


### Calculate additional parameters of orbit ###
def calc_params():
    """ Input the period and print out additional info on the respective orbit.
    >> period
    >> eccentricity
    >> radius extrema
    >> Perihelion distance
    >> Semimajor axis
    >> Semiminor axis
    >> total energy
    """
    print("====== ORBIT RESULTS ======")
    print(f"orbits={orbits}, runs={runs}, dt={dt}\n") # simulation parameters

    # print period
    print("period =", round(period, 4), "[yr]")

    # total energy E = K + (-P)
    E = 0.5*m*v_0**2 - (GM*m)/r_0 # using pre-calculated G*M
    print("total energy E =", round(E, 4), "[M AU^2 / yr^2]") #TODO unit of energy

    # Semimajor axis
    sma = (r_max + r_min) / 2 # mean value of the r extrema
    print("semimajor axis =", round(sma,4), "[AU]")

    # Semiminor axis
    smi = math.sqrt(r_max * r_min) # geometric mean of the r extrema
    print("semiminor axis =", round(smi,4), "[AU]")

    # Measured e
    e_measured2 = math.sqrt(1 - (smi **2 / sma **2))
    print("eccentricity (measured) e =", round(e_measured2,4))

    # radius extrema
    print("\nr_max =", round(r_max,4), "[AU]")
    print("r_min =", round(r_min,4), "[AU]")

    # Perihelion distance P=sma*(1-e)
    print("perihelion distance =", round( (sma*(1-e_measured2)),4), "[AU]")
    print("---------------------------------------------")

####################### RUN SIMULATION ########################

if confirmKeplersThirdLaw:
    """ Plots several orbits. Then plots graph to confirm Kepler's Third Law """
    # NEW KEPLER Experiment
    orbits = 1
    runs = 1
    r_0 = 1
    v_0 = 2*pi
    dt = dt # default 0.005
    kep_range = 20
    v_list = np.linspace(1.8*pi, 2.5*pi, kep_range).tolist()
    r0 = 1 # initial radius 1 [AU]
    #print(f"r_list:\n{r_list}\n") #DEBUGGING
    #print(f"v_list:\n{v_list}\n") #DEBUGGING

    # Compute periods and sma for input values
    period_list = []  # list of periods
    sma_list = []  # list of semi-major axis values
    for i in range(kep_range):
        v0 = v_list[i]
        r_max, r_min, period = compute_orbit(r0, v0)
        period_list.append(period)  # store period length
        sma_list.append((r_max + r_min) / 2) # store semi-major axes
        plot_polar()

    period_list_plot = [p**2 for p in period_list] # square entries of period_list
    sma_list_plot = [sma**3 for sma in sma_list] # cube entries of sma_list
    print("Plot all computed orbits to show that they are stable ...")
    plt.draw() # show keplerplot
    #plt.savefig("plot_kepler_orbits")

    # close all plots at once
    plt.pause(1) # pause to make time for the input line
    answer = input(f"Close all {kep_range} plots? [Y]es or [N]o? ").lower()
    if answer in ('yes','y'):
        for i in range(kep_range):
            plt.close()
    else:
        answer = input(f"Press [Enter] to continue...")
        for i in range(kep_range):
            plt.close()

    #print(f"period_list_plot:\n{period_list_plot}\n") #DEBUGGING
    #print(f"sma_list_plot:\n{sma_list_plot}\n") #DEBUGGING
    plot_kepler(period_list_plot, sma_list_plot)  # plot graph to visually confirm Kepler's Third Law

else:
    """ standard plot of some orbit """
    # normally compute position
    r_max, r_min, period = compute_orbit(r_0, v_0)
    calc_params() # print out simulation parameters
    plot_position() # plot position over time
    #plot_angle()   # used to visualize orbit length during debugging
    plot_polar() # plot angular position of orbit
    plot_energy()
