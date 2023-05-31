# hexanematic

Cite this code: Armengol-Collado et al. 2023 (Nature Physics)
Author: Livio Nicola Carenza
Date: 2023-5 

These Python scripts determine the p-fold shape parameter starting from cell segmentation and coarse-grain the orientation field over a given length-scale. 
Topological defects are identified and nematic/hexatic graphical reconstruction produced in output.

Script 'shape_function.py': this program computes the shape function gamma_p (for p=2,...,6) for each cell in a given configuration as defined in Eq. 2 of the above reference, and print them to a file.

Script 'coarse_graining.py': this program coarse-grains the shape functions, gamma_p, over a length-scale R for a number of values specified in the program parameters. The resulting p-fold shape parameter, Gamma_p, (see Eq.(3) in the above Reference) is averaged over different configurations.

Script 'plot_nematics.py': this program identifies nematic topological defects in a given configuration and produce a graphical output of the nematic reconstruction.

Script 'plot_hexatics.py': this program identifies hexatic topological defects in a given configuration and produce a graphical output of the hexatic reconstruction.

Read the description at the beginning of each python script for a detailed description of input/output files and program parameters.
