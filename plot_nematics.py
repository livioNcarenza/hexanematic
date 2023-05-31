# ------ Coarse-graining the p-fold shape function ------
# 
# Cite this code: Armengol-Collado et al. 2023 (Nature Physics)
#
# Author: Livio Nicola Carenza
# Date: 2023-5 
#
#
# --- DESCRIPTION: 
# This program identifies nematic topological defects in a given configuration and produce a graphical output of the nematic reconstruction 
#
#
# --- INPUT: 
# This program requires in input text files 'Gamma2_grid_cgR_$$_config_##.dat', where $$ stands for the coarse-graining radius used to perform the coarse-graining 
# and ## stands for the index of the configuration. These files are produced in output by the program 'coarse_graining.py'.
#
# The input files must be formatted as follows.
# Each line must contain 4 columns: 
# - the x coordinate 
# - the y coordinate
# - the magnitude of Gamma2 at position (x,y)
# - the orientation of Gamma2 at position (x,y)
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
# The program produce a graphical output 'nematic.svg' showing a color plot of the cross-polarizer texture of the nematic field.
# The defects are superimposed to the color plot.
# Nematic director can also be plotted. Comment lines 194-202 if not wanted.
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
p2=2.0

deltax = 21.51480   # set this equal to the x-step of the coarse-graining grid
deltay = 21.51480   # set this equal to the y-step of the coarse-graining grid


LLarrx = 22*deltax  # set this equal to the x-size of the coarse-graining grid
LLarry = 22*deltay  # set this equal to the y-size of the coarse-graining grid

xx = np.linspace(deltax, LLarrx, int(LLarrx/deltax))
yy = np.linspace(deltax, LLarry, int(LLarry/deltay))

Psi2theta = np.zeros((len(xx),len(yy)))
#######################################################################################

#######################################################################################
#reading configuration
#######################################################################################
filename='Gamma2_grid_cgR_3_config_3.dat'        
data = np.genfromtxt(filename)
x = data[:,0]
y = data[:,1]
theta2 = data[:,3]
	
for i in range(len(xx)):
	for j in range(len(yy)):
		Psi2theta[i][j] = theta2[j+i*len(yy)]
#######################################################################################

#######################################################################################
#This routine detects topological defects
#######################################################################################
imod=[0]*4
jmod=[0]*4
phi2=[0]*4
delta2=[0]*4
defect2_xpos=[]   # this list will contain the x-coordinate of defects' position
defect2_ypos=[]   # this list will contain the y-coordinate of defects' position
defect2_q=[]      # this list will contain the charge of identified defects


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
			
		phi2[0] = Psi2theta[imod[0]][jmod[0]]
		phi2[1] = Psi2theta[imod[1]][jmod[1]]
		phi2[2] = Psi2theta[imod[2]][jmod[2]]
		phi2[3] = Psi2theta[imod[3]][jmod[3]]
		
		delta2[0] = phi2[1]-phi2[0]
		delta2[1] = phi2[2]-phi2[1]
		delta2[2] = phi2[3]-phi2[2]
		delta2[3] = phi2[0]-phi2[3]
		
		winding_number2 = 0.0

		for n in range(len(delta2)):
			if (delta2[n] <=- PI/p2):
				delta2[n] = delta2[n] + 2.0*PI/p2
			if (delta2[n] > PI/p2):
				delta2[n] = delta2[n] - 2.0*PI/p2
			winding_number2 += delta2[n]
		if(abs(winding_number2)>0.01):
			defect2_xpos.append(0.5*(2*xx[i]+1))
			defect2_ypos.append(0.5*(2*yy[j]+1))
			defect2_q.append(winding_number2/(2.0*PI))

pdef2_x=[]
pdef2_y=[]
ndef2_x=[]
ndef2_y=[]	

for dd in range (len(defect2_xpos)):
	if (defect2_q[dd]>0):
		pdef2_x.append(defect2_xpos[dd])
		pdef2_y.append(defect2_ypos[dd])				
	else:
		ndef2_x.append(defect2_xpos[dd])
		ndef2_y.append(defect2_ypos[dd])
		
pdef2_x=np.array(pdef2_x)
pdef2_y=np.array(pdef2_y)
ndef2_x=np.array(ndef2_x)
ndef2_y=np.array(ndef2_y)
#######################################################################################

#######################################################################################
# plot configuration
#######################################################################################
# figure initialization
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif":	["Helvetica"], "font.size": 12})
fig, (ax2) = plt.subplots(nrows=1, ncols=1)

### plotting nematic director (comment if not wanted)
scale=15
for i in range(len(xx)):
	for j in range(len(yy)):
		angle=float(Psi2theta[i][j])
		Nx = cos(angle)
		Ny = sin(angle)
		Nx2 = -Nx
		Ny2 = -Ny
		ax2.arrow(xx[i],LLarry-yy[j],scale*Nx, -scale*Ny,  head_width=0.0,head_length=0., color='#8AC926')
		ax2.arrow(xx[i],LLarry-yy[j],scale*Nx2,-scale*Ny2, head_width=0.0,head_length=0., color='#8AC926')
###

### plotting cross-polarizer texture
X, Y = np.meshgrid(xx, yy)
Z2=np.transpose(Psi2theta)
Z2=2*abs(np.cos(Z2)**2-0.5)
norm2 = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
pcm = ax2.imshow(Z2, cmap='binary', extent=[min(xx), max(xx), min(yy), max(yy)], aspect=1, interpolation = 'none', norm=norm2)
fig.colorbar(pcm, ax=ax2, shrink=.7, location='top', orientation='horizontal')

### plotting defects
ax2.scatter(pdef2_x-deltax/2, LLarry-pdef2_y+deltay/2, marker='o', s=25, color='red',  edgecolors='white')   #positive
ax2.scatter(ndef2_x-deltax/2, LLarry-ndef2_y+deltay/2, marker='o', s=25, color='blue', edgecolors='white')   #negative

### figure settings
plt.tick_params(axis='x',which='both') 
plt.tick_params(axis='y',which='both') 
ax2.set_xlim(min(xx),max(xx))
ax2.set_ylim(min(yy),max(yy))
ax2.set_aspect(1)

### saving figure
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'nematic.svg'
fig.savefig(image_name, format=image_format, dpi=1200)
plt.close()



