# ------ Coarse-graining the p-fold shape function ------
# 
# Cite this code: Armengol-Collado et al. 2023 (Nature Physics)
#
# Author: Livio Nicola Carenza
# Date: 2023-5 
#
#
# --- DESCRIPTION: 
# This program coarse-grains the shape functions, gamma_p, (see Eq.(2) in 
# the above Reference) over a length scale R for a number of values specified in the program parameters. 
# The resulting p-fold shape parameter, Gamma_p, (see Eq.(3) in the above Reference) is averaged over different configurations.
#
#
# --- INPUT: 
# This program requires in input text files 'exp_#.dat', where # stands for a number 
# ranging from 1 to the total number of configurations to analyze.
#
# Each file is formatted as follows.
# Each line refers to a cell in the configuration. 
# The first column (indexed 0) contains the x-coordinate of the centre of mass (com) of the cell. 
# The second column (indexed 1) contains the y-coordinate of the centre of mass (com) of the cell. 
# The third column (indexed 2) contains the real part of the complex shape function gamma_2 of the cell. 
# The fourth column (indexed 3) contains the imaginary part of the complex shape function gamma_2 of the cell. 
# The fifth column (indexed 4) contains the real part of the complex shape function gamma_3 of the cell and so on.
#
# --- PARAMETERS: 
#
# cell_radius:  is the average cell radius measured in experiments. Here is reported in pixel units. 
# minR_CG:      is the smallest coarse-graining radius (in cell radius units)
# maxR_CG:      is the largest coarse-graining radius (in cell radius units) 
# stepR_CG:     is the step between two coarse-grining radii (in cell radius units) 
# LLarr:        size of image in pixel units
#
#
# --- OUTPUT:
# 
# The program prints text files 'Gammap_vs_radius.dat' in output (for p=2,...,6). 
# Each output file is formatted as follows.
# Each lines has three columns containing 
# -the coarse-graining radius,
# -the averaged magnitude of the p-atic order paramter
# -the standard deviation
#
#
# The program can also produce as output the grid of coarse-grained order parameter (see lines 267--287)
# The output files are named 'Gamma2_grid_cgR_CCGGRR_config_CCFFGG.dat' 
# where CCGGRR is a number standing for coarse-graining radius used to perform the coarse-graining and CCFFGG is the index of the configuration.
# These data files have four columns containing:
# - the x coordinate 
# - the y coordinate
# - the magnitude of Gammap at position (x,y)
# - the orientation of Gammap at position (x,y)
# ------------------------------------------------------

import matplotlib.pyplot
import matplotlib.mlab
import math
import numpy as np
from pylab import *
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings
import seaborn as sns
import scipy as sp
import scipy.fftpack
import readchar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.ticker
from scipy.stats import norm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import interpolate
from scipy.optimize import curve_fit
from pathlib import Path

##############################################
### Function Definition ###
##############################################
def pwr_law(x, A, B):
    y = A*(x**B)
    return y

def dist_bc(pt1x,pt2x,pt1y,pt2y,box_lengthx,box_lengthy):
	deltax = abs(pt1x-pt2x)
	deltay = abs(pt1y-pt2y)
	if (deltax > box_lengthx-deltax):
		deltax = box_lengthx - deltax
	if (deltay > box_lengthy-deltay):
		deltay = box_lengthy - deltay
	dd = np.sqrt(deltax**2 + deltay**2)
	return dd

def dist(pt1x,pt2x,pt1y,pt2y):
	deltax = abs(pt1x - pt2x)
	deltay = abs(pt1y - pt2y)
	dd = np.sqrt(deltax**2 + deltay**2)
	return dd
##############################################
##############################################

##############################################
### PARAMETERS Definition ###
##############################################
PI = 4*math.atan(1)   

cell_radius = 21.5  # cell radius (here set in pixel units)
minR_CG=1           # minimum coarse-graining radius (in units of cell radius)
maxR_CG=15          # maximum coarse-graining radius (in units of cell radius)
stepR_CG=1          # step between coarse-graining radius (in units of cell radius)  

#creating list of coarse-graining radii (these are stored in the array RR)
RR=[]
for radius in range(minR_CG,maxR_CG,stepR_CG):
    RR.append(radius)
RR=np.array(RR)

LLarr = 500                                       #defining size of coarse-graining grid (in pixel units)
xx = np.linspace(0, LLarr, 23, endpoint=False)    #defining x-coordinates of coarse-graining grid (in pixel units)
yy = np.linspace(0, LLarr, 23, endpoint=False)    #defining y-coordinates of coarse-graining grid (in pixel units)

#these lists will be use to store the magnitude of the p-atic order parameter 
#at a given coarse graining radius *for each* configuration analyzed
Gamma2_vs_RR=[]
Gamma3_vs_RR=[]
Gamma4_vs_RR=[]
Gamma5_vs_RR=[]
Gamma6_vs_RR=[]
##############################################
##############################################


##############################################
### Body of the program ###
##############################################

iter_analyzed=0
maxIter = 68
#the 'for' cycle iterates over all the experimental dataset (see program details on the top for a description of the input file)
for iteration in np.arange(1,maxIter+1,1):
    
  print(iteration)
  iter_analyzed += 1

  #read input file
  file2open='exp_'
  file2open=file2open+str(iteration)+'.dat'

  data = np.genfromtxt(file2open)       
  Xcm = data[:,0]         #Xcm contains the x-coordinates (in pixel units) of the centre of mass of all cells in the configuration 
  Ycm = data[:,1]         #Ycm contains the y-coordinates (in pixel units) of the centre of mass of all cells in the configuration
  gammaR2 = data[:,2]     #gammaR2 contains the real component of the shape function gamma_2 of all cells in the configuration
  gammaI2 = data[:,3]     #gammaI2 contains the imaginary component of the shape function gamma_2 of all cells in the configuration
  gammaR3 = data[:,4]     #...
  gammaI3 = data[:,5]     #...
  gammaR4 = data[:,6]
  gammaI4 = data[:,7]
  gammaR5 = data[:,8]
  gammaI5 = data[:,9]
  gammaR6 = data[:,10]
  gammaI6 = data[:,11]

  #these lists will be use to store the magnitude of the p-atic order parameter 
  #at a given coarse graining radius *for a single* configuration
  Gamma2_vs_radius=[]
  Gamma3_vs_radius=[]
  Gamma4_vs_radius=[]
  Gamma5_vs_radius=[]
  Gamma6_vs_radius=[]


  #the 'for' cycle iterates over all coarse-graining radii (expressed in units of cell radius)
  for radius_c in RR:

    radius = radius_c*cell_radius                 #defining cell radius in pixel units

    Gamma2Mag = np.zeros((len(xx),len(yy)))       #initializing the magnitude of the shape parameter Gamma_2 at each sampling position
    Gamma3Mag = np.zeros((len(xx),len(yy)))       #initializing the magnitude of the shape parameter Gamma_3 at each sampling position
    Gamma4Mag = np.zeros((len(xx),len(yy)))       #...
    Gamma5Mag = np.zeros((len(xx),len(yy)))       #...
    Gamma6Mag = np.zeros((len(xx),len(yy)))
    
    Gamma2theta = np.zeros((len(xx),len(yy)))     #initializing the orientation of the shape parameter Gamma_2 at each sampling position
    Gamma3theta = np.zeros((len(xx),len(yy)))     #initializing the orientation of the shape parameter Gamma_3 at each sampling position
    Gamma4theta = np.zeros((len(xx),len(yy)))     #...
    Gamma5theta = np.zeros((len(xx),len(yy)))     #...
    Gamma6theta = np.zeros((len(xx),len(yy)))
    
    
    #the 'for' cycle iterates over all sampling points
    for i in range(len(xx)):
      for j in range(len(yy)):
      
        #initializing variables for coarse-graining at position (xx[i], yy[j])
        sum_Rp2=0.0
        sum_Ip2=0.0

        sum_Rp3=0.0
        sum_Ip3=0.0

        sum_Rp4=0.0
        sum_Ip4=0.0

        sum_Rp5=0.0
        sum_Ip5=0.0

        sum_Rp6=0.0
        sum_Ip6=0.0

        counter=0
        #the 'for' cycle iterates over all cells in the configuration
        for cell in range (len(gammaR2)):
          # computing distance (in pixel units) between the centre of mass of the cell with respect to the sampling point (xx[i], yy[j])
          distance = dist(xx[i],Xcm[cell],yy[j],Ycm[cell])                      
          # if analyzing simulation data with periodic boundary conditions comment previous line and use the following instead:
          # distance = dist_bc(xx[i],Xcm[cell],yy[j],Ycm[cell],LLarr,LLarr)

          #cell contribute to coarse-graining only if closer than coarse-graining radius
          if distance <= radius:
            counter += 1
            sum_Rp2 += gammaR2[cell]    
            sum_Ip2 += gammaI2[cell]

            sum_Rp3 += gammaR3[cell]
            sum_Ip3 += gammaI3[cell]

            sum_Rp4 += gammaR4[cell]
            sum_Ip4 += gammaI4[cell]

            sum_Rp5 += gammaR5[cell]
            sum_Ip5 += gammaI5[cell]

            sum_Rp6 += gammaR6[cell]
            sum_Ip6 += gammaI6[cell]

        #deviding over the number of cells inside the coarse-graining radius
        if (counter):
        	sum_Rp2 = sum_Rp2/counter
        	sum_Ip2 = sum_Ip2/counter

        	sum_Rp3 = sum_Rp3/counter
        	sum_Ip3 = sum_Ip3/counter

        	sum_Rp4 = sum_Rp4/counter
        	sum_Ip4 = sum_Ip4/counter

        	sum_Rp5 = sum_Rp5/counter
        	sum_Ip5 = sum_Ip5/counter

        	sum_Rp6 = sum_Rp6/counter
        	sum_Ip6 = sum_Ip6/counter

        #defining shape parameter Gamma_p at position (xx[i], yy[j])
        Gamma2Mag[i][j]=np.sqrt(sum_Rp2*sum_Rp2+sum_Ip2*sum_Ip2)
        Gamma2theta[i][j]=1.0/2.0*(math.atan2(sum_Ip2,sum_Rp2))

        Gamma3Mag[i][j]=np.sqrt(sum_Rp3*sum_Rp3+sum_Ip3*sum_Ip3)
        Gamma3theta[i][j]=1.0/3.0*(math.atan2(sum_Ip3,sum_Rp3))

        Gamma4Mag[i][j]=np.sqrt(sum_Rp4*sum_Rp4+sum_Ip4*sum_Ip4)
        Gamma4theta[i][j]=1.0/4.0*(math.atan2(sum_Ip4,sum_Rp4))

        Gamma5Mag[i][j]=np.sqrt(sum_Rp5*sum_Rp5+sum_Ip5*sum_Ip5)
        Gamma5theta[i][j]=1.0/5.0*(math.atan2(sum_Ip5,sum_Rp5))

        Gamma6Mag[i][j]=np.sqrt(sum_Rp6*sum_Rp6+sum_Ip6*sum_Ip6)
        Gamma6theta[i][j]=1.0/6.0*(math.atan2(sum_Ip6,sum_Rp6))
        
        #################### end of cycle over sampling points ####################
    
    #####################################################################################################################
    # comment the following lines if output of coarse-grained grid is not wanted
    # a text file containing the coarse-grained shape parameter will be printed for each state and each coarse-graining radius
    #####################################################################################################################
    #output grid (p=2)
    #filename='Gamma2_grid_cgR_'+str(radius_c)+'_config_'+str(iteration)+'.dat'
    #file2write=open(filename,'w')
    #for i in range(len(xx)):
    #  for j in range(len(yy)):
    #    file2write.write("%E %E %E %E\n" % (xx[i], yy[j], Gamma2Mag[i][j], Gamma2theta[i][j]))
    #file2write.close()
    
    #output grid (p=6)
    #filename='Gamma6_grid_cgR_'+str(radius_c)+'_config_'+str(iteration)+'.dat'
    #file2write=open(filename,'w')
    #for i in range(len(xx)):
    #  for j in range(len(yy)):
	  #    file2write.write("%E %E %E %E\n" % (xx[i], yy[j], Gamma6Mag[i][j], Gamma6theta[i][j]))
    #file2write.close()
    #####################################################################################################################
    #####################################################################################################################
    
    
    #averaging the shape parameter Gamma_p over the whole configuration
    counter = 0
    avg2 = 0
    avg3 = 0
    avg4 = 0
    avg5 = 0
    avg6 = 0

    for i in range(len(xx)):
      for j in range(len(yy)):
        if(Gamma2Mag[i][j]>0):
          counter += 1
          avg2 += Gamma2Mag[i][j]
          avg3 += Gamma3Mag[i][j]
          avg4 += Gamma4Mag[i][j]
          avg5 += Gamma5Mag[i][j]
          avg6 += Gamma6Mag[i][j]

    if (counter > 0):
      avg2 /= counter
      avg3 /= counter
      avg4 /= counter
      avg5 /= counter
      avg6 /= counter

    #the average value of the shape parameter Gamma_p for a specific coarse-graining radius is added to the list Gamma2_vs_radius
    Gamma2_vs_radius.append(avg2)
    Gamma3_vs_radius.append(avg3)
    Gamma4_vs_radius.append(avg4)
    Gamma5_vs_radius.append(avg5)
    Gamma6_vs_radius.append(avg6)
    
    #################### end of cycle over different coarse-graining radii ####################
    
  ##########################################################################################
  ### Gammap_vs_radius now contains the average value of Gamma_p computed for a single configuration at different coarse-graining radii
  ##########################################################################################

  Gamma2_vs_radius=np.array(Gamma2_vs_radius)
  Gamma3_vs_radius=np.array(Gamma3_vs_radius)
  Gamma4_vs_radius=np.array(Gamma4_vs_radius)
  Gamma5_vs_radius=np.array(Gamma5_vs_radius)
  Gamma6_vs_radius=np.array(Gamma6_vs_radius)

  ### Gammap_vs_radius is appended to the list Gamma2_vs_RR
  Gamma2_vs_RR.append(Gamma2_vs_radius)
  Gamma3_vs_RR.append(Gamma3_vs_radius)
  Gamma4_vs_RR.append(Gamma4_vs_radius)
  Gamma5_vs_RR.append(Gamma5_vs_radius)
  Gamma6_vs_RR.append(Gamma6_vs_radius)

  #################### end of cycle over different configurations ####################

Gamma2_vs_RR=np.array(Gamma2_vs_RR)
Gamma3_vs_RR=np.array(Gamma3_vs_RR)
Gamma4_vs_RR=np.array(Gamma4_vs_RR)
Gamma5_vs_RR=np.array(Gamma5_vs_RR)
Gamma6_vs_RR=np.array(Gamma6_vs_RR)

#defining variables for avarages over different configurations
avGamma2=np.zeros(len(RR))
avGamma3=np.zeros(len(RR))
avGamma4=np.zeros(len(RR))
avGamma5=np.zeros(len(RR))
avGamma6=np.zeros(len(RR))

errGamma2=np.zeros(len(RR))
errGamma3=np.zeros(len(RR))
errGamma4=np.zeros(len(RR))
errGamma5=np.zeros(len(RR))
errGamma6=np.zeros(len(RR))

#computing the mean of the shape parameter over different configurations (at same coarse-graining radius)
for rad in range (len(RR)):
  for iteration in range(iter_analyzed):
    avGamma2[rad] +=  Gamma2_vs_RR[iteration,rad]/iter_analyzed
    avGamma3[rad] +=  Gamma3_vs_RR[iteration,rad]/iter_analyzed
    avGamma4[rad] +=  Gamma4_vs_RR[iteration,rad]/iter_analyzed
    avGamma5[rad] +=  Gamma5_vs_RR[iteration,rad]/iter_analyzed
    avGamma6[rad] +=  Gamma6_vs_RR[iteration,rad]/iter_analyzed


#computing the standard deviation
for rad in range (len(RR)):
    for iteration in range(iter_analyzed):
        errGamma2[rad] += (Gamma2_vs_RR[iteration,rad]-avGamma2[rad])**2/iter_analyzed
        errGamma3[rad] += (Gamma3_vs_RR[iteration,rad]-avGamma3[rad])**2/iter_analyzed
        errGamma4[rad] += (Gamma4_vs_RR[iteration,rad]-avGamma4[rad])**2/iter_analyzed
        errGamma5[rad] += (Gamma5_vs_RR[iteration,rad]-avGamma5[rad])**2/iter_analyzed
        errGamma6[rad] += (Gamma6_vs_RR[iteration,rad]-avGamma6[rad])**2/iter_analyzed

errGamma2=np.sqrt(errGamma2)
errGamma3=np.sqrt(errGamma3)
errGamma4=np.sqrt(errGamma4)
errGamma5=np.sqrt(errGamma5)
errGamma6=np.sqrt(errGamma6)



### PRINTING TO FILE
file2write=open("Gamma2_vs_radius.dat",'w')
for i in range(len(RR)):
	file2write.write("%E %E %E\n" % (RR[i],avGamma2[i], errGamma2[i]))
file2write.close()

file2write=open("Gamma3_vs_radius.dat",'w')
for i in range(len(RR)):
	file2write.write("%E %E %E\n" % (RR[i],avGamma3[i], errGamma3[i]))
file2write.close()

file2write=open("Gamma4_vs_radius.dat",'w')
for i in range(len(RR)):
	file2write.write("%E %E %E\n" % (RR[i],avGamma4[i], errGamma4[i]))
file2write.close()

file2write=open("Gamma5_vs_radius.dat",'w')
for i in range(len(RR)):
	file2write.write("%E %E %E\n" % (RR[i],avGamma5[i], errGamma5[i]))
file2write.close()

file2write=open("Gamma6_vs_radius.dat",'w')
for i in range(len(RR)):
	file2write.write("%E %E %E\n" % (RR[i],avGamma6[i], errGamma6[i]))
file2write.close()
#######################################################################################
