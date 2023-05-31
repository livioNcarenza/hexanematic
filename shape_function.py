# ------ p-fold shape function ------
# 
# Cite this code: Armengol-Collado et al. 2023 (Nature Physics)
#
# Author: Livio Nicola Carenza
# Date: 2023-5 
#
#
# --- DESCRIPTION: 
# This program computes the shape function gamma_p (for p=2,...,6) 
# as defined in Eq. 2 of the above reference, and print them to a file.
# 
#
# --- INPUT: 
# This program accepts as input two different kind of data.
# CASE 1: the input are the coordinates of the vertices of the segmented cells. 
#   In this case the input data files must be organized as follows:
#   1st line: x-coordinates of 1st segmented cell 
#   2nd line: y-coordinates of 1st segmented cell 
#   3rd line: x-coordinates of 2nd segmented cell 
#   4th line: y-coordinates of 2nd segmented cell 
#   (2n-1)-th line: x-coordinates of n-th segmented cell 
#   (2n)-th line: y-coordinates of n-th segmented cell 
#   
# Notice that:
# - the number of coordinates in a pair of consecutive lines (odd-even) must be equal
# (to the number of vertices of a given cell). Failing to format data in this way leads the program to crash.
# - the separator between vertices coordinates in each line must be a simple space ' ' (see lines 178 and 181).
# 
#
# CASE 2: the input are the centers of the segmented cells. 
#   In this case each line in the input data file corresponds to a single cell.
#   Each lines has two columns containing:
#   - the x-coordinate of the cell's center 
#   - the y-coordinate of the cell's center 
#   
# Notice that:
# - In the above reference, CASE 1 was used to compute the shape function gamma_p of the single cells.
# - You can switch between CASE 1 and CASE 2 by opportunely setting the switch 'input_vertex' equal to 1 or 2, respectively.
#   (see line 135)
# 
#
# --- PARAMETERS: 
# LLarrx:        # system size (x-direction)
# LLarry:        # system size (y-direction)
#
#
# --- OUTPUT:
# The program prints text files 'gammap_singlecell.dat' in output. 
# Each output file is formatted as follows.
# Each lines has 12 columns containing 
# col. 1: the x-coordinate of the cell's center, 
# col. 2: the y-coordinate of the cell's center, 
# col. 3: the real component of the cell's shape function gamma_2, 
# col. 4: the imaginary component of the cell's shape function gamma_2, 
# col. 5: the real component of the cell's shape function gamma_3, 
# col. 6: the imaginary component of the cell's shape function gamma_3, 
# ...
# col. 11: the real component of the cell's shape function gamma_6, 
# col. 12: the imaginary component of the cell's shape function gamma_6.
#
# These output files can be used as input files for the python program 'coarse_graining.py'.
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
import matplotlib.ticker
from scipy.stats import norm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import interpolate
from scipy.optimize import curve_fit

plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif":	["Helvetica"], "font.size": 12})

##############################################
### Function Definition ###
##############################################
def cm_periodic(vert,cell,LLarr):
	x1=0.0
	x2=0.0
	vv = vert[cell]
	for pt in range(0,len(vv)):
		x1 += cos(2.0*np.pi*vv[pt]/(LLarr-1))
		x2 += sin(2.0*np.pi*vv[pt]/(LLarr-1))
	cm_coord = (LLarr-1)*(math.atan2(-x2,-x1)+np.pi)/(2.0*np.pi)
	return cm_coord
	
def cm_periodic_from_point_list(points,LLarr):
	x1=0.0
	x2=0.0
	for pt in range(0,len(points)):
		x1 += cos(2.0*np.pi*points[pt]/(LLarr-1))
		x2 += sin(2.0*np.pi*points[pt]/(LLarr-1))

	cm_coord = (LLarr-1)*(math.atan2(-x2,-x1)+np.pi)/(2.0*np.pi)
	return cm_coord
	
def cord_bc(pt1,pt2,box_length):
	delta = pt1-pt2
	if (delta > box_length-delta):
		delta = box_length - delta
	if (delta < delta - box_length):
		delta = box_length + delta
	return delta
	
def cord(pt1,pt2):						
	dd = pt1 - pt2
	return dd
	
def dist_bc(pt1x,pt2x,pt1y,pt2y,box_lengthx,box_lengthy):						
	deltax = abs(pt1x - pt2x)
	deltay = abs(pt1y - pt2y)
	if (deltax > box_lengthx-deltax):
		deltax = box_lengthx - deltax
	if (deltay > box_lengthy-deltay):
		deltay = box_lengthy - deltay
	dd = np.sqrt(deltax**2 + deltay**2)
	return dd
	


##############################################
##############################################

PI = 4*math.atan(1)
input_vertex = 1                                # set this switch to 1 if the input data are the vertices of cell segmentation
                                                # otherwise, if the input data are the centers of the cells, set the switch to 2        

LLarrx = 144                                    # system size (x-direction)
LLarry = 124                                    # system size (y-direction)

Xcm=[]            # this will be used to store the x-coordinate of the center of the cells
Ycm=[]            # this will be used to store the y-coordinate of the center of the cells
Xcm_pr=[]            # this will be used to store the x-coordinate of the center of the cells
Ycm_pr=[]            # this will be used to store the y-coordinate of the center of the cells
vertx=[]          # this will be used to store the x-coordinate of the vertices of the cells
verty=[]          # this will be used to store the y-coordinate of the vertices of the cells

gamma2_R=[]       # this will be used to store the real component of the shape function gamma_2 of each cell
gamma2_I=[]       # this will be used to store the imaginary component of the shape function gamma_2 of each cell

gamma3_R=[]       # this will be used to store the real component of the shape function gamma_3 of each cell
gamma3_I=[]       # this will be used to store the imaginary component of the shape function gamma_3 of each cell

gamma4_R=[]       # ...
gamma4_I=[]       # ...

gamma5_R=[]
gamma5_I=[]

gamma6_R=[]
gamma6_I=[]

################################################################################################################
# READ INPUT FILES

################################################################################################################ 
#### CASE 1: input files are the position of the vertices. No Voronoi tessellation needed.
################################################################################################################
if (input_vertex == 1):
  file2open='vertices.dat'
  V_x=[]
  V_y=[]

  data_vertices = open(file2open)
  counterline=0
  for line in data_vertices.readlines():
    if ( (counterline % 2) == 0 ):
      vvx = np.fromstring(line, dtype=float, sep=' ')
      vertx.append(vvx)
    else:
      vvy = np.fromstring(line, dtype=float, sep=' ')
      verty.append(vvy)
    counterline += 1
  data_vertices.close()
  
  for cell in range (len(vertx)):
    Xcm.append(cm_periodic(vertx,cell,LLarrx))
    Ycm.append(cm_periodic(verty,cell,LLarry))

################################################################################################################
################################################################################################################

################################################################################################################ 
#### CASE 2: input files are the position of particles. Voronoi tessellation needed to find the vertices.
#### COMMENT/DECOMMENT THE FOLLOWING LINE IF YOU ARE USING CASE 2/CASE 1
################################################################################################################
if (input_vertex == 2):
  file2open='centers.dat'
  data_centres = np.genfromtxt(file2open)         
  xctr = data_centres[:,1]                        # xctr contains the x-positions of all particles in the configuration
  yctr = data_centres[:,2]                        # yctr contains the y-positions of all particles in the configuration
  xctr=np.array(xctr)
  yctr=np.array(yctr)

  points = np.array([xctr,yctr])                  # the array 'points' contains both x and y coordinate of the centre of mass of all cells
  points = points.transpose()

  #compute voronoi tessellation
  vor = Voronoi(points)

  #this cycle computes the center of each voronoi polygon and stores it in
  for region in vor.regions:
	  xcm=0.0
	  ycm=0.0
	  vvx=[]
	  vvy=[]
	  #this checks if the polygon is out of tessellation
	  if -1 not in region and region:
		  index = vor.regions.index(region)
		  for pt in region:
		    vvx.append(vor.vertices[pt][0])
		    vvy.append(vor.vertices[pt][1])
		    xcm += vor.vertices[pt][0]
		    ycm += vor.vertices[pt][1]
		  xcm = xcm/len(region)
		  ycm = ycm/len(region)
		  Xcm.append(xcm)
		  Ycm.append(ycm)
		  vertx.append(vvx)
		  verty.append(vvy)
################################################################################################################
################################################################################################################


for cell in range (len(Xcm)):

  norm2=0.0
  psiR2=0.0
  psiI2=0.0

  norm3=0.0
  psiR3=0.0
  psiI3=0.0

  norm4=0.0
  psiR4=0.0
  psiI4=0.0

  norm5=0.0
  psiR5=0.0
  psiI5=0.0

  norm6=0.0
  psiR6=0.0
  psiI6=0.0

  for pt in range(len(vertx[cell])):
    
    xpr = cord(vertx[cell][pt], Xcm[cell])
    ypr = cord(verty[cell][pt], Ycm[cell])
    # if the data have periodic boundary conditions (the have been obtained 
    # through simulations) comment previous lines and use these one instead.
    #xpr = cord_bc(vertx[cell][pt], Xcm[cell], LLarrx)
    #ypr = cord_bc(verty[cell][pt], Ycm[cell], LLarry)
    
    d_i = np.sqrt(xpr*xpr+ypr*ypr)
    tth = math.atan2(ypr,xpr)

    norm2 += d_i**2
    psiR2 += (d_i**2)*cos(2.0*tth)
    psiI2 += (d_i**2)*sin(2.0*tth)

    norm3 += d_i**3
    psiR3 += (d_i**3)*cos(3.0*tth)
    psiI3 += (d_i**3)*sin(3.0*tth)

    norm4 += d_i**4
    psiR4 += (d_i**4)*cos(4.0*tth)
    psiI4 += (d_i**4)*sin(4.0*tth)

    norm5 += d_i**5
    psiR5 += (d_i**5)*cos(5.0*tth)
    psiI5 += (d_i**5)*sin(5.0*tth)

    norm6 += d_i**6
    psiR6 += (d_i**6)*cos(6.0*tth)
    psiI6 += (d_i**6)*sin(6.0*tth)

  if(norm2):
    psiR2 = psiR2/norm2
    psiI2 = psiI2/norm2
    gamma2_R.append(psiR2)
    gamma2_I.append(psiI2)
    # if the magnitude and the angle of gamma_p are needed these can be 
    # respectively obtained as follows:
    #modgamma_p2.append(np.sqrt(psiR2*psiR2+psiI2*psiI2))
    #theta2.append(1.0/2.0*(math.atan2(psiI2,psiR2)))

  if(norm3):
    psiR3 = psiR3/norm3
    psiI3 = psiI3/norm3
    gamma3_R.append(psiR3)
    gamma3_I.append(psiI3)

  if(norm4):
    psiR4 = psiR4/norm4
    psiI4 = psiI4/norm4
    gamma4_R.append(psiR4)
    gamma4_I.append(psiI4)

  if(norm5):
    psiR5 = psiR5/norm5
    psiI5 = psiI5/norm5
    gamma5_R.append(psiR5)
    gamma5_I.append(psiI5)

  if(norm6):
    psiR6 = psiR6/norm6
    psiI6 = psiI6/norm6
    gamma6_R.append(psiR6)
    gamma6_I.append(psiI6)


### PRINTING TO FILE
file2write=open("gammap_singlecell.dat",'w')
for i in range(len(gamma2_R)):
  file2write.write("%E %E %E %E %E %E %E %E %E %E %E %E\n" % (Xcm[i], Ycm[i], gamma2_R[i], gamma2_I[i], gamma3_R[i], gamma3_I[i], gamma4_R[i], gamma4_I[i], gamma5_R[i], gamma5_I[i], gamma6_R[i], gamma6_I[i]))
file2write.close()

