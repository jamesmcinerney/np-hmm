'''
Animation of GPS track data;  takes a .csv file from the GPS device, computes (x = Long., y = Lat. in meters, z = elevation) vs. time (sec) and acceleration
  then animates them
feel free to use, but email me first:  samador@haverford.edu
and cite:  Suzanne Amador Kane 11-15-11  http://www.haverford.edu/physics-astro/Amador/index.php
  
Suzanne Amador Kane & Emma Oxford, Haverford College Physics Dept.
 
This program takes as an input .csv files created by your GPS unit (or other sensors, such as altimeters for altitude)
digests this data, then converts GPS (Longitude, Latitude) to (x,y) units in meters (checked against online calculators)
then plots GPS coordinates along withe vectors showing magnitude and direction of velocity and acceleration on an
animation of the moving GPS receiver using visual python, and saves all this data into new .dat files.
Uses a Gaussian smoothing filter with sigma = 0.4 s to smooth the GPS coordinate data.
 
OUTPUTS:
      NOTE:  Because current GPS altitude (elevation) measurements are so unreliable, we have commented out any z-calculations.
             If you have an accurate altimeter installed, alter the program to read in this data and use it for the z-values as indicated.
             All z-calculations are as shown, they just need to be activated.
      Creates a .dat file with the same corename as the input .csv file that holds all computed values in SI units (sec, meters, etc.):
             time, x, y, velocity magnitude), vel.x, vel.y, acceleration (magnitude), a_tan.x, a_tan.y, a_cen.x, a_cen.y,Gaussian curvature
             where .x, .y indicate the (x,y) components, while cen = centripetal and tan = tangential accelerations.

Build instructions:  I am running all this on a Windows 7 machine using the following python build:
  
  python 2.7.1              http://www.python.org/getit/releases/2.7/
  visual python 2.7a0       http://vpython.org/index.html
  numpy-1.5.1-py2.7         http://sourceforge.net/projects/numpy/files/NumPy/1.5.1/
  matplotlib-1.0.1-py2.7    http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.0/
  scipy-0.9.0-py2.7         http://sourceforge.net/projects/scipy/files/scipy/0.9.0/

USING IT:
 
INPUTS:  Requires a .csv file.  Since many GPS units do not provide this, we often convert .gpx or other common GPS format files
           into .csv files using the freeware GPS software BT747, available at:  http://www.bt747.org/
           You will need to study your resulting output files to find out which column your GPS coordinates are stored in, and which
           your altimeter data (if any) are stored in.   See *** below.
           
HOW TO SEE YOUR TRACKS ON A REAL MAP OR SATELLITE PHOTO
    If you don't know already, you can do both using Google Earth.
    Using GPS Visualizer (a free online utility at http://www.gpsvisualizer.com/), you can even create 3D tracks if you also have
    accurate altitude data and do lots of other interesting things. (Current GPS technology cannot provide good altitude readings, but new real time kinetic methods ought to soon make this available at reasonable cost.
    Stay tuned!

LIMITATIONS:
    GPS data is currently limited (as of 2012) by wavelength issues (and hence timing difference resolution) that give it an
    accuracy and precision no better than about 2.5 m for most models.  Real time kinetic GPS (which uses the phase information)
    can deliver centimeter-resolution readings along all three Cartesian directions, but it's currently too bulk and expensive for
    uses--but stay tuned!  Trimble and Novetel now make reasonably small RTK GPS units and the technology will advance fast.

EQUIPMENT SUGGESTIONS:
    We've used Qstarz GPS dataloggers (the nano variety), Vernier Instruments portable loggers with their GPS and barometric sensor sensor
    (acting as an altimeter), and Eagle Tree Systems eLogger V4 datalogger with their GPS, altimeter and other sensors.  The Qstarz
    and Eagle Tree systems are very lightweight and possible to mount on even small animals.  The logging rates at 5 Hz for the Qstarz (20 grams)
    and 10 Hz (Eagle tree;  about 45 gram).  Even ligther weight GPS dataloggers can be found that log at only 1 Hz, down to about 15 gram.

Ideas for improvements:  See *** below for any places where we might improve things.  Did you do any of this?  Send us your code please!
  1)  Add in z (elevation or altitude) data from an altimeter.  This is not accurately logged by the GPS, and the Vernier device doesn't even record it
      due to the inherent problems in finding elevation from satellites mostly near the horizon.
  2)  Change the center of view dynamically so you can zoom in and view the different parts of the trajectories--see *** below
  3)  Use matplotlib to import a map image and overlay the trajectory on it, though this is done so nicely already in Google (and within LoggerPro), but you can't
      see the combined map and acceleration/velocity vectors that way.  Too bad!
      How to at:  http://stackoverflow.com/questions/6347537/overlaying-a-scatter-plot-on-background-image-and-changing-axes-ranges and
      http://matplotlib.sourceforge.net/users/image_tutorial.html
  4)  Why not add more sensors?  Accelerometers can provide faster sampling, better acceleration measurents, orientational information
  (if stationary with respect to the Earth's surface) and allow interpolation between GPS position readings.  IMU's (inertial
  measurement units--basically MEMS gyros) can provide orientational data.  And there are others--airspeed, etc.  See Eagle Tree Systems
  and other vendors (Sparkfun, DIYdrones.com, etc.) for more ideas!

'''

import numpy
import math
from visual import *
from visual.controls import *
from visual.graph import * # import graphing features
from visual.filedialog import get_file

###### initialization section ##########################################################################################

# datalogging setup
f_s = 5.0                           # *** sampling frequency in Hz (1 for Vernier;  5 for Qstarz GPS units)
dt = 1/f_s                          # sampling period in seconds
time_index = 0                      # *** the index of the time in the input .csv file (column 1 so 0 for Qstarz data logger, for Vernier, 0)
Lat_index = 8                       # *** the index of the latitude in the input .csv file (column 9 so 8 for Qstarz data logger, for Vernier, 4)
Long_index = 10                     # *** the index of the longitude in the input .csv file (column 11 so 10 for Qstarz data logger, for Vernier, 5)
Long_sign = 1                       # *** set = 1 if the longitude is indicated by an explicit sign;  otherwise, use =-1 for W, 1 for E
# Elev_index = 6                      #*** the index of the altitude in the input .csv file (? for Qstarz data logger, for Vernier, 6)

playback_speed = 600                # *** number of updates per second in animation--set at your GPS logging rate in Hz for real time

n_plot = 8                          # *** how many time steps to wait before plotting velocity and acceleration
a_magnify = 30                      # *** an arbitrary factor:  how much to magnify our acceleration vectors to make them visible

# GPS (Long., Lat.) to [meters] setup
# I used an online calculator for GPS coordinates (several actually) and got results that differ from Emma's in the
# 4th sig fig, but all the online calculator agreed.  Use these values instead:
##print "conversion factor deg to m lat at long 75.304683",333.24992/(40.005922-40.00892095)
##print "conversion factor deg to m long at lat  ",155.18820/(75.304683-75.30650631)
# center of parking lot:  lat = 40.005922; long = 75.304683
# right stone wall outside Hilles lat = 40.00892095; long =75.30650631
long_to_m = 155.18820/(75.304683-75.30650631)
lat_to_m = -333.24992/(40.005922-40.00892095)

gaussian_filter_flag = 1            # *** set to 1 for Gaussian filtering;  0 to turn this feature off
# creating Gaussian filter array to smooth data
# if you have scipy, you can use the canned module there, but some of our computers lack it.
if gaussian_filter_flag:
    sigma = 0.40                                        # sigma = 0.4 s, same as Nagy et al. Nature 2010 pigeon hierarchy GPS methods
    gaussian_filter = []
    filter_num=6
    for n in arange(1,2*filter_num+2,1):# For sigma = 0.4 sec and 5Hz sampling rate, the Gaussian goes below 1% at filter_num=6 points to each side
        n_filter = n-(filter_num+1)     # set the center of the array equal to zero
        gaussian_filter.append(exp(-(n_filter*0.2)*(n_filter*0.2)/(2.*sigma*sigma) ))
else:
    gaussian_filter = ones(2*filter_num+2)

gaussian_filter = gaussian_filter/sum(gaussian_filter)  # normalize the gaussian weighting factor

######################################################################################################
# get a file with actual measured GPS trajectory data
# 
import os.path
import csv 

# file i/o section:  read in the .csv file created by the BT747 (or other) GPS program
# and digest it to get (time, x,y and possibly z), then compute the acceleration vector vs. time
# 

print "Get GPS data now (as .CSV file)"
fd = get_file()

if fd:
    dataA = fd.read()                           
    fd.close()                                  # close the file (we're through with it)
print fd.name
    
ifile  = open(fd.name, "rb")
reader = csv.reader(ifile)

datatemp=[]
row_num = 0                                     # this variable keeps a tally of how many rows are in your file;  later we need to subtract 1 for the header
for row in reader:
    # Save header row.
    if row_num == 0:                            # remove if your GPS files do not have a header
        header = row
    else:
        for n in arange(len(row)):
            datatemp.append(row[n])
    row_num += 1

ifile.close()

#only some of the entries are real numbers;  others are strings, so we have to parse this up in order to figure out its values

GPSdata = array(datatemp)

num_columns = len(row)                          # number of columns should be number in each row;  rownum = # rows + 1
row_num=row_num - 1

num_frames = row_num                            # number of time samples in our GPS data

GPSdata.shape = row_num,num_columns    

#-------------------------------------------------------------------------------
# Write data results as data file in format:  time,x,y,z using Flat Earth conversions
# Time is dt*index (element 0 in each row); x can be computed from Lat ( Lat_index) and Long ()Long_index, elevation = ?
# GPS_coords will have 0 = time, 1 = x and 2 = y (in seconds, meters and meters) (and elevation in meters if defined from altimeter data
GPS_coords = zeros((row_num,4))
for i in arange(num_frames):
    GPS_coords[i][0]=dt*float(GPSdata[i][0])                                        # time in seconds assuming sampling period of dt = 1/f_s (sampling rate)
    GPS_coords[i][3]= 0.                                                            # elevation is zero for now;  later get from altimeter
    if 6 <= i and i <= num_frames-7:                                                # enough points to Gaussian smooth the input data
        for n in arange(1,2*filter_num+2,1):
            n_filter = n-(filter_num+1)                                             # computes where we are in the Gaussian filter list relative to the center at filter_num + 1
            GPS_coords[i][1]= GPS_coords[i][1] +gaussian_filter[n-1]* long_to_m*(Long_sign*float(GPSdata[i+n_filter][Long_index])-75.30650631)
                                                                                    # Longitude (+ = east, - = west;  all our data should be negative for now)
            GPS_coords[i][2]= GPS_coords[i][2]+ gaussian_filter[n-1]* lat_to_m*(float(GPSdata[i+n_filter][Lat_index])-40.00892095)
                                                                                    # Latitude (+ = northern hemisphere where we are  of course)

    else:                                                                           # not enough points--just copy unsmoothed data from input file
        GPS_coords[i][1]= long_to_m*(Long_sign*float(GPSdata[i][Long_index])-75.30650631)     # Longitude (+ = east, - = west;  all our data should be negative for now)
        GPS_coords[i][2]= lat_to_m*(float(GPSdata[i][Lat_index])-40.00892095)       # Latitude (+ = northern hemisphere where we are  of course)


######################################################################################################################
# Write data results as data file

GPS_coords_file = os.path.abspath(fd.name)                              # get name of input file

GPS_coords_file_corename=os.path.splitext(GPS_coords_file)              # get only the core (no extension) of input file for use in naming output files

GPS_coords_name = GPS_coords_file_corename[0]+".dat"                    # create a new file that will hold the analyzed data

fd_coords = open(GPS_coords_name, 'w')

######################################################################################################################
# plot out a trajectory of the actual flight trajectory of the object tracked via GPS

# find the maximum and minimum dimensions along x and y for plotting:
x_max = 1.05*max(GPS_coords[:,1])
x_min = 1.05*min(GPS_coords[:,1])
y_max = 1.05*max(GPS_coords[:,2])
y_min = 1.05*min(GPS_coords[:,2])
z_max = 0. # *** set if you have the altimeter data
z_min = 0. # *** set if you have the altimeter data

max_arrow_length = 0.1*(x_max-x_min)    # *** our animated arrows can't get plotted if they are bigger than this
                                        # just keeps the vectors from dominating the animation
# set the scale of the entire animation scene and plot a bounding wireframe

grey = (0.2,0.2,0.2)    # color of the wireframe

display(title = "GPS data analysis",width=600, height=600*(y_max-y_min)/(x_max-x_min)) # create and center the display window for the animation
scene.autocenter = False #turn autocenter off

scene_origin = vector((x_max+x_min)/2.,(y_max+y_min)/2.,(z_max+z_min)/2.) # center the display window on our plot center
x_max = (x_max-x_min)/2.
x_min = - x_max
y_max = (y_max-y_min)/2.
y_min = - y_max
z_max = (z_max-z_min)/2.
z_min = - z_max

square1 = curve(pos=[(x_max,y_max,z_max),(x_min,y_max,z_max),(x_min,y_min,z_max),(x_max,y_min,z_max),(x_max,y_max,z_max)],color=grey)
square2 = curve(pos=[(x_max,y_max,z_min),(x_min,y_max,z_min),(x_min,y_min,z_min),(x_max,y_min,z_min),(x_max,y_max,z_min)],color=grey)
square3 = curve(pos=[(x_max,y_max,z_max),(x_max,y_max,z_min),(x_max,y_min,z_min),(x_max,y_min,z_max),(x_max,y_max,z_max)],color=grey)
square4 = curve(pos=[(x_min,y_max,z_max),(x_min,y_max,z_min),(x_min,y_min,z_min),(x_min,y_min,z_max),(x_min,y_max,z_max)],color=grey)

GPS_start = vector(GPS_coords[0,1],GPS_coords[0,2],GPS_coords[0,3])-scene_origin
GPS_receiver = sphere(pos=GPS_start,radius=0.01*(x_max-x_min),color=color.yellow,make_trail=True)
######################################################################################################################

t_0 = t = GPS_coords[0,0]                                       # initial time in the video

accel = zeros(num_frames)                                       # acceleration magnitude array initialization

for i in arange(num_frames):
    rate (playback_speed)                                       # refresh rate--30 fps is usual video rate
    t = GPS_coords[i,0] - t_0                                   # real world time in seconds(minus initial frame time t_0)
    
    # find velocities for each region near ith frame time
    # only fits when we have enough data on either side of this time index
    delta_n = 2     # this is now half the window for computing the velocity : interval = 2*delta_n + 1
    # inspect raw positional data to see over what interval we have well-defined slopes and curvatures to determine this value
    # this value of 3 gives 7 data points, about 1.4 sec which seems fine for computing velocities upon inspection
    if i >= delta_n and i < num_frames - (delta_n + 1):         # in the middle range of data set--OK to do fit
        t_array = GPS_coords[i-delta_n:i+delta_n+1,0]
        x_array = GPS_coords[i-delta_n:i+delta_n+1,1]
        y_array = GPS_coords[i-delta_n:i+delta_n+1,2]
        z_array = GPS_coords[i-delta_n:i+delta_n+1,3]
        # now find velocity by fitting a line to x vs. t, y vx. t, z vs. t:
        # this section does a tidy nonlinear least squares fit to a polynomial
        # first, form the Vandermonde matrix with the independent data array, t_array
        degree = 2 # degree of polynomial = 2 for a line
        A_coeff = vander(t_array, degree)
        # find the coefficients that minimizes the norm of A_coeff,x_array(the dependent variable array)
        (coeffs, residuals, rank, sing_vals) = linalg.lstsq(A_coeff, x_array)
        vx = coeffs[0]
        (coeffs, residuals, rank, sing_vals) = linalg.lstsq(A_coeff, y_array)
        vy = coeffs[0]
        (coeffs, residuals, rank, sing_vals) = linalg.lstsq(A_coeff, z_array)
        vz = coeffs[0]
        velocity = vector(vx,vy,vz) 
        ### end of velocity analysis for actual tracking data
    elif i < num_frames -1 :                                    # can't do a fit, so just compute the difference for the velocity
        velocity=vector(GPS_coords[i+1,1]-GPS_coords[i,1],GPS_coords[i+1,2]-GPS_coords[i,2],
                               GPS_coords[i+1,3]-GPS_coords[i,3])/(GPS_coords[i+1,0]-GPS_coords[i,0])

    ############################################################### START new section to compute acceleration
    # use polynomial fit here to get acceleration in each direction and, use same delta_n_a to compute radius of curvature
    delta_n_a = 2     # this is now half the window for computing the acceleration: interval = 2*delta_n_a + 1
    # inspect raw positional data to see over what interval we have well-defined slopes and curvatures to determine this value
    # given how smooth our data is, we ought to be able to get away with this small a value of delta_n_a = 4 or 5 and still get constant velocity slopes
    # over most time intervals;  this should capture the constant accelerations well
    # Note that Vernier uses a total of 7 points for smoothing (delta_n_a=3)
    if i >= delta_n_a and i < num_frames - (delta_n_a + 1):
        # only fits when we have enough data on either side of this time index
        made_it = 1
        # slice up the arrays first
        t_array = GPS_coords[i-delta_n_a:i+delta_n_a+1,0]
        x_array = GPS_coords[i-delta_n_a:i+delta_n_a+1,1]
        y_array = GPS_coords[i-delta_n_a:i+delta_n_a+1,2]
        z_array = GPS_coords[i-delta_n_a:i+delta_n_a+1,3]
        # now, find acceleration
        # this section does a tidy nonlinear least squares fit to a polynomial
        # first, form the Vandermonde matrix with the independent data array, t_array
        degree = 3 # degree of polynomial = 3 for a parabola

        Aa_coeff = vander(t_array, degree)
        # find the coefficients that minimizes the norm of Aa_coeff,x_array(the dependent variable array)
        (coeffs, residuals, rank, sing_vals) = linalg.lstsq(Aa_coeff, x_array)
        # return the slope (velocity) (1) and curvature (acceleration) (0)
        ax = 2*coeffs[0]
        (coeffs, residuals, rank, sing_vals) = linalg.lstsq(Aa_coeff, y_array)
        # return the slope (velocity) (1) and curvature (acceleration) (0)
        ay = 2*coeffs[0]
        (coeffs, residuals, rank, sing_vals) = linalg.lstsq(Aa_coeff, z_array)
        # return the slope (velocity) (1) and curvature (acceleration) (0)
        az = 2*coeffs[0]
        accel[i]=(mag(vector(ax,ay,az)))

        #radius of curvature: need arc length and chord between points P and Q
        P = vector(GPS_coords[i-delta_n_a][1],GPS_coords[i-delta_n_a][2],GPS_coords[i-delta_n_a][3])
        Q = vector(GPS_coords[i+delta_n_a+1][1],GPS_coords[i+delta_n_a+1][2],GPS_coords[i+delta_n_a+1][3])
        chord = mag(Q-P)

        arc_length = 0.
        for j in arange(2*delta_n_a+1):  # *** added 1 here since you're leaving out the last point and hence Q--that's why arc_length < chord!
            initial_point = vector(GPS_coords[i-delta_n_a+j][1],GPS_coords[i-delta_n_a+j][2],GPS_coords[i-delta_n_a+j][3])
            final_point = vector(GPS_coords[i-delta_n_a+j+1][1],GPS_coords[i-delta_n_a+j+1][2],GPS_coords[i-delta_n_a+j+1][3])

            arc_length = arc_length + mag(final_point - initial_point)
        if arc_length != 0. and arc_length > chord:
            curvature = sqrt((24*(arc_length - chord))/pow(arc_length,3))
        else:
            curvature = 0.
    else:
        curvature = 0.
        arc_length = 0.
        chord = 0.
        accel[i]= 0
        ax = 0
        ay = 0
        az = 0

    acceleration_vector = vector(ax,ay,az)
    accel_magnitude = mag(vector(ax,ay,az))
    vel_magnitude = mag(velocity)
    normalize_velocity = norm(velocity)
    magnitude_tangential = dot(acceleration_vector,normalize_velocity)
    tangential_acceleration = magnitude_tangential*normalize_velocity
    centripetal_acceleration = acceleration_vector - tangential_acceleration
    magnitude_centripetal = mag(centripetal_acceleration)

    # Write to a file t, x, y, #z, vel.mag, vel.x, vel.y, a_mag, a_tan.x, a_tan.y, a_cen.x, a_cen.y, curvature

    fd_coords.write(str(GPS_coords[i,0])+" ")           #time
    fd_coords.write(str(GPS_coords[i,1])+" ")           #longitude (x)
    fd_coords.write(str(GPS_coords[i,2])+" ")           #latitude (y)
    #fd_coords.write(str(GPS_coords[i,3])+" ")          #altitude (z) writes x,y,z coordinates for each track, in order, on the same line for each frame
    fd_coords.write(str(vel_magnitude)+" ")             #velocity magnitude
    fd_coords.write(str(velocity.x)+" ")
    fd_coords.write(str(velocity.y)+" ")                #velocity vector
    fd_coords.write(str(accel_magnitude)+" ")           #total acceleration magnitude
    fd_coords.write(str(magnitude_tangential)+" ")      #tangential acceleration vector
    fd_coords.write(str(magnitude_centripetal)+" ")     #centripetal acceleration vector
    fd_coords.write(str(curvature)+" ")                 #Gaussian curvature
    fd_coords.write("\n")                               #newline only after each frame is finished

    ############################################################### END compute acceleration
                                                        # plot animated GPS receiver in right position
    GPS_receiver.pos = vector(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z)

    # we have to magnify our various vectors to see them on the same scale as the trajectory and velocities
    ax = a_magnify*ax
    ay = a_magnify*ay
    az = a_magnify*az
    tangential_acceleration = 5.*tangential_acceleration
    centripetal_acceleration = 5.*centripetal_acceleration
    if i== 0:
        acceleration_vector = vector(ax,ay,az)
        normalize_velocity = norm(velocity)
        magnitude_tangential = dot(acceleration_vector,normalize_velocity)
        tangential_acceleration = magnitude_tangential*normalize_velocity
        centripetal_acceleration = acceleration_vector - tangential_acceleration
        velocity_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(velocity.x,velocity.y,velocity.z), color=color.green,opacity=0.5)
        acceleration_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(ax,ay,az), color=color.red,opacity=0.5)
        tangential_acceleration_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(tangential_acceleration.x,tangential_acceleration.y,tangential_acceleration.z), color=color.cyan)
        centripetal_acceleration_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(centripetal_acceleration.x,centripetal_acceleration.y,centripetal_acceleration.z), color=color.magenta)
        # In addition to velocity and acceleration vectors, it is also useful to know how much of the acceleration is due to banking (centripetal) and how much is due to speeding up and slowing down (tangential).
    if i > n_plot-1 and i%n_plot == 1 :                            # only plot velocity and acceleration once every 2 seconds
        acceleration_vector = vector(ax,ay,az)
        normalize_velocity = norm(velocity)
        magnitude_tangential = dot(acceleration_vector,normalize_velocity)
        tangential_acceleration = magnitude_tangential*normalize_velocity
        centripetal_acceleration = acceleration_vector - tangential_acceleration
        if velocity.mag <= max_arrow_length:
            velocity_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(velocity.x,velocity.y,velocity.z), color=color.green,opacity=0.5)
        if acceleration_vector.mag <= max_arrow_length:
            acceleration_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(ax,ay,az), color=color.red,opacity=0.5)
        if tangential_acceleration.mag <= max_arrow_length:
            tangential_acceleration = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(tangential_acceleration.x,tangential_acceleration.y,tangential_acceleration.z), color=color.cyan)
        if centripetal_acceleration.mag <= max_arrow_length:
            centripetal_acceleration_arrow = arrow(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z), axis=(centripetal_acceleration.x,centripetal_acceleration.y,centripetal_acceleration.z), color=color.magenta)
        # In addition to velocity and accelration vectors, it is also useful to know how much of the acceleration is due to banking (centripetal) and how much is due to speeding up and slowing down (tangential).
            sphere(pos=(GPS_coords[i,1]-scene_origin.x,GPS_coords[i,2]-scene_origin.y,GPS_coords[i,3]-scene_origin.z),radius=0.005*(x_max-x_min),color=color.yellow)

fd_coords.close()
print "green   arrows = velocity"
print "red     arrows = acceleration"
print "cyan    arrows = tangential acceleration"
print "magenta arrows = centripetal acceleration"
print "blue   spheres = position every ",n_plot*dt," sec"