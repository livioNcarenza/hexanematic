# ------ Coarse-graining the p-fold shape function ------
# 
# Cite this code: Armengol-Collado et al. 2023 (Nature Physics)
#
# Author: Livio Nicola Carenza
# Date: 2023-5 
#
#
# --- DESCRIPTION: 
# This program identifies hexatic topological defects in a given configuration and produce a graphical output of the hexatic reconstruction 
#
#
# --- INPUT: 
# This program requires in input text files 'Gamma6_grid_cgR_$$_config_##.dat', where $$ stands for the coarse-graining radius used to perform the coarse-graining 
# and ## stands for the index of the configuration. These files are produced in output by the program 'coarse_graining.py'.
#
# The input files must be formatted as follows.
# Each line must contain 4 columns: 
# - the x coordinate 
# - the y coordinate
# - the magnitude of Gamma6 at position (x,y)
# - the orientation of Gamma6 at position (x,y)
#
#
# --- PARAMETERS: 
# deltax:  x-step of the coarse-graining grid
# deltay:  y-step of the coarse-graining grid
# LLarrx:  x-size of the coarse-graining grid
# LLarry:  y-size of the coarse-graining grid
# Notice that these values must correspond to those used to generate the configuration in the program 'coarse_graining.py'.
#
#
# --- OUTPUT:
# The program produce a graphical output 'hexatic.svg' showing a color plot of the hexatic director's angle.
# The defects are superimposed to the color plot.
# Hexatic director (6 legged stars) can also be plotted. Comment lines 192-214 if not wanted.
# ------------------------------------------------------


import matplotlib.pyplot
import numpy as np
from pylab import *

##############################################
### Function Definition ###
##############################################
def cord_bc(pt1,pt2,box_length):
	delta = pt1-pt2
	if (delta > box_length-delta):
		delta = box_length - delta
	if (delta < delta - box_length):
		delta = box_length + delta
	return delta

def dist_bc(pt1x,pt2x,pt1y,pt2y,box_lengthx,box_lengthy):
	deltax = cord_bc(pt1x,pt2x,box_lengthx)
	deltay = cord_bc(pt1y,pt2y,box_lengthy)
	if (deltax > box_lengthx-deltax):
		deltax = box_lengthx - deltax
	if (deltay > box_lengthy-deltay):
		deltay = box_lengthy - deltay
	dd = np.sqrt(deltax**2 + deltay**2)
	return dd
##############################################
##############################################

##############################################
### PARAMETERS Definition ###
##############################################
PI = 4*math.atan(1)
p6=6.0

deltax = 21.51480   # set this equal to the x-step of the coarse-graining grid
deltay = 21.51480   # set this equal to the y-step of the coarse-graining grid

LLarrx = 22*deltax  # set this equal to the x-size of the coarse-graining grid
LLarry = 22*deltay  # set this equal to the x-size of the coarse-graining grid

xx = np.linspace(deltax, LLarrx, int(LLarrx/deltax))
yy = np.linspace(deltax, LLarry, int(LLarry/deltay))

Psi6theta = np.zeros((len(xx),len(yy)))
#######################################################################################

#######################################################################################
#reading configuration
#######################################################################################
filename='Gamma6_grid_cgR_3_config_8.dat'        
data = np.genfromtxt(filename)
x = data[:,0]
y = data[:,1]
theta6 = data[:,3]
	
for i in range(len(xx)):
	for j in range(len(yy)):
		Psi6theta[i][j] = theta6[j+i*len(yy)]
#######################################################################################

#######################################################################################
#This routine detects topological defects
#######################################################################################
imod=[0]*4
jmod=[0]*4
phi6=[0]*4
delta6=[0]*4
defect6_xpos=[]   # this list will contain the x-coordinate of defects' position
defect6_ypos=[]   # this list will contain the y-coordinate of defects' position
defect6_q=[]      # this list will contain the charge of identified defects


### CASE 1: experimental images (no periodic boundary)
for i in range(1,len(xx)):
	for j in range(1,len(yy)):
		imod[0]=i
		imod[1]=i-1
		imod[2]=i-1
		imod[3]=i	

		jmod[0]=j
		jmod[1]=j
		jmod[2]=j-1	
		jmod[3]=j-1	

### CASE 2: simulation images (WITH periodic boundary)
#for i in range(0,len(xx)):
#	for j in range(0,len(yy)):
#		imod[0]=i
#		imod[1]=i-1
#		if (i==0): imod[1]=len(xx)-1
#		imod[2]=i-1
#		if (i==0): imod[2]=len(xx)-1
#		imod[3]=i	
#
#		jmod[0]=j
#		jmod[1]=j
#		jmod[2]=j-1	
#		if (j==0): jmod[2]=len(yy)-1
#		jmod[3]=j-1	
#		if (j==0): jmod[2]=len(yy)-1
			
		phi6[0] = Psi6theta[imod[0]][jmod[0]]
		phi6[1] = Psi6theta[imod[1]][jmod[1]]
		phi6[2] = Psi6theta[imod[2]][jmod[2]]
		phi6[3] = Psi6theta[imod[3]][jmod[3]]
		
		delta6[0] = phi6[1]-phi6[0]
		delta6[1] = phi6[2]-phi6[1]
		delta6[2] = phi6[3]-phi6[2]
		delta6[3] = phi6[0]-phi6[3]
		
		winding_number6 = 0.0

		for n in range(len(delta6)):
			if (delta6[n] <=- PI/p6):
				delta6[n] = delta6[n] + 2.0*PI/p6
			if (delta6[n] > PI/p6):
				delta6[n] = delta6[n] - 2.0*PI/p6
			winding_number6 += delta6[n]
		if(abs(winding_number6)>0.01):
			defect6_xpos.append(0.5*(2*xx[i]+1))
			defect6_ypos.append(0.5*(2*yy[j]+1))
			defect6_q.append(winding_number6/(2.0*PI))

pdef6_x=[]
pdef6_y=[]
ndef6_x=[]
ndef6_y=[]	

for dd in range (len(defect6_xpos)):
	if (defect6_q[dd]>0):
		pdef6_x.append(defect6_xpos[dd])
		pdef6_y.append(defect6_ypos[dd])				
	else:
		ndef6_x.append(defect6_xpos[dd])
		ndef6_y.append(defect6_ypos[dd])
		
pdef6_x=np.array(pdef6_x)
pdef6_y=np.array(pdef6_y)
ndef6_x=np.array(ndef6_x)
ndef6_y=np.array(ndef6_y)
#######################################################################################

#######################################################################################
# plot configuration
#######################################################################################
# figure initialization
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif":	["Helvetica"], "font.size": 12})
orig_map=plt.cm.get_cmap('hsv')
reversed_map = orig_map.reversed()
fig, (ax2) = plt.subplots(nrows=1, ncols=1)

### plotting hexatic star (comment if not wanted)
#scale=6
#for i in range(len(xx)):
#  for j in range(len(yy)):
#    angle=float(Psi6theta[i][j])
#    Hx = cos(angle)
#    Hy = sin(angle)
#    Hx2 = Hx*cos(2.0*PI/6.0) - Hy*sin(2.0*PI/6.0)
#    Hy2 = Hx*sin(2.0*PI/6.0) + Hy*cos(2.0*PI/6.0)	
#    Hx3 = Hx2*cos(2.0*PI/6.0) - Hy2*sin(2.0*PI/6.0)
#    Hy3 = Hx2*sin(2.0*PI/6.0) + Hy2*cos(2.0*PI/6.0)
#    Hx4=-Hx
#    Hy4=-Hy
#    Hx5=-Hx2
#    Hy5=-Hy2
#    Hx6=-Hx3
#    Hy6=-Hy3
#    ax2.arrow(xx[i],yy[j],scale*Hx, scale*Hy,  head_width=0.0,head_length=0.)
#    ax2.arrow(xx[i],yy[j],scale*Hx2,scale*Hy2, head_width=0.0,head_length=0.)
#    ax2.arrow(xx[i],yy[j],scale*Hx3,scale*Hy3, head_width=0.0,head_length=0.)
#    ax2.arrow(xx[i],yy[j],scale*Hx4,scale*Hy4, head_width=0.0,head_length=0.)
#    ax2.arrow(xx[i],yy[j],scale*Hx5,scale*Hy5, head_width=0.0,head_length=0.)
#    ax2.arrow(xx[i],yy[j],scale*Hx6,scale*Hy6, head_width=0.0,head_length=0.)

### plotting cross-polarizer texture
X, Y = np.meshgrid(xx, yy)
Z6=np.transpose(Psi6theta)+PI/6.0
norm6 = matplotlib.colors.Normalize(vmin=0.0, vmax=PI/3.0,)
pcm = ax2.imshow(Z6, cmap=reversed_map, extent=[min(xx), max(xx), min(yy), max(yy)], aspect=1, interpolation = 'none', norm=norm6)
fig.colorbar(pcm, ax=ax2, shrink=.7, location='top', orientation='horizontal')

### plotting defects
ax2.scatter(pdef6_x-deltax/2, LLarry-pdef6_y+deltay/2, marker='o', s=25, color='red',  edgecolors='white')   #positive
ax2.scatter(ndef6_x-deltax/2, LLarry-ndef6_y+deltay/2, marker='o', s=25, color='blue', edgecolors='white')   #negative

### figure settings
plt.tick_params(axis='x',which='both') 
plt.tick_params(axis='y',which='both') 
ax2.set_xlim(min(xx),max(xx))
ax2.set_ylim(min(yy),max(yy))
ax2.set_aspect(1)

### saving figure
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'hexatic.svg'
fig.savefig(image_name, format=image_format, dpi=1200)
plt.close()
