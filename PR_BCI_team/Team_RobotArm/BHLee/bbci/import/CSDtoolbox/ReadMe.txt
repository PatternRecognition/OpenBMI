CSD Toolbox
===========
© 2003-2009 by Jürgen Kayser

Version 1.0 (May 22, 2009)

The CSD toolbox provides a MatLab implementation of a spherical spline algorithm (Perrin et al., 1989) to compute scalp surface Laplacian or current source density (CSD) estimates for surface potentials (EEG/ERP). Its successful use requires basic knowledge of (or willingness to learn about) electrophysiologic principles (e.g., EEG montage, 10-20-system, volume conduction) and rudimentary mathematics (e.g., fundamental algebra and geometry), as well as access to and basic knowledge of MatLab® software. Without these prerequisites, the use of the CSD toolbox is strongly discouraged.

The CSD Toolbox archive includes the following files:

 ..\ReadMe.txt                                    this file
 ..\doc\History.txt                               detailed documentation of changes
 ..\ : \WhatsNew.txt                              synopsis of most important revisions
 ..\exam\ExampleCode1.m                           example code used for online tutorial
 ..\ :  \E31.asc                                  example EEG montage used in online tutorial
 ..\ :  \NR_C66_trr.dat                           example ERP data used in online tutorial
 ..\ :  \GetF3CoordinatesWithMirrorLocations.m    example code showing how to compute 10-20-system spherical coordinates
 ..\func\ConvertLocations.m                       routine to convert between EEGlab and CSD toolbox spherical coordinates
 ..\ :  \ExtractMontage.m                         routine to extract an EEG montage from a *.csd coordinates look-up table
 ..\ :  \MapMontage.m                             routine to display a 2-D topography of an EEG montage
 ..\ :  \CSD.m                                    routine to compute CSD estimates from surface potentials (EEG/ERP)
 ..\ :  \GetGH.m                                  routine to compute transformation matrices G and H used with CSD.m
 ..\ :  \WriteMatrix2Text.m                       routine to write a data matrix to an ASCII text file
 ..\ :  \LineWithSphere.m                         routine to compute the sphere surface intersection(s) of a line defined by two 3-D points
 ..\ :  \SphericalMidPoint.m                      routine to compute a point on an unit sphere with equal distance to two spherical points
 ..\resource\10-5-System_Mastoids_EGI129.csd      look-up table for 330 standard 10-20-system and 129 geodesic (GSN) scalp locations

Extract the content of the CSD toolbox zip archive to a local folder and add the folder with its subfolders to the Matlab path.

For documentation, tutorial, FAQ, common errors, and general information and references, see the CSD Toolbox home page at:
 
 *****************************************************************
 * http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox *
 *****************************************************************

This software is intended for non-profit scientific research. It is copyright-protected under the GNU General Public License (see agreement at http://www.gnu.org/licenses/gpl.txt).