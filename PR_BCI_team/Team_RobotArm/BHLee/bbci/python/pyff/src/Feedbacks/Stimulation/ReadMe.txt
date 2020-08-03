-----------------------------------------------------------------
GENERAL INFORMATION

1. Required Installations:
- Glyph package http://pygame.org/project-Glyph-1002-2794.html
- pyAudiere package http://pyaudiere.org/
- pygame package http://pygame.org/download.shtml

2. setup_relax.py derives from pygamefeedback and contains general methods:
- to read text,
- translate latex format to pygame display format,
- display text,
- show fixation cross (either single central or multiple),
- read and play sounds,
- show countdown and
- create the animation.

3. stim_artifactMeasurement.py contains the running code for eyes open/ closed condition
- init_ToBeSetBeforeExec() need to be set by the User before the run
- self.seq controlls the presentation sequence
Eg: self.seq = 'P2000 F21P2000 R[10] (F14P15000 F1P2000 F15A15000 F1P2000) F20P1000'. See setup_relax.parse for details.

4. functions.py contains some user defined functions.

5. Translate.py translates latex type rendering into Glyph readable format. 
- Modify the init_glyph to add more fonts and modes to the display. Currently it uses Freesans and Freesansbold only. Eg: add :Macros['i']=('font',fontfile)
- Then modify the txt file to be displayed accordingly

-----------------------------------------------------------------
INSTALLING GLYPH

1. Download and Unpack the package from the site mentioned earlier.

2. The unpacked folder (example: glpyh-2.5.1) then has a sub-folder named glyph along with files LICENSE.txt, README.txt, setup.py and PKG-INFO. Copy these files into the sub-folder glyph.

3. Then copy and paste the glyph folder in the Python directory  ../Python26/Lib/site-packages/

4. Python should now be able to use and import Glyph.
-----------------------------------------------------------------