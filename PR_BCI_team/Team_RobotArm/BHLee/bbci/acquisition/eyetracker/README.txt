

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SMI RED-250 Eyetracker Matlab Interface %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



|| For 64-bit systems, Matlab 8 or higher is required (problems with c-compiler for older matlab versions)


|| The communication between matlab and the iViewX software is done via dynamic-link-libraries (DLL) that are loaded into matlab.
   For more information on the functions available, see acquisition\eyetracker\iViewXAPI\Docs\iView X SDK Manual.pdf


|| For older matlab versions, there are differences in the header file of the DLL library. This is accounted for when 
   smi('init') is called from matlab, but you might want to check it in detail in case problems arise at this stage.


|| If running the iViewX program from another computer than the original SMI laptop, take care that

	- when installing the RED driver (from the SMI iViewX CD), do not connect the RED-250 via USB until you're told to
	  do so in the command line.

	- the RED-250 is connected at a USB 2.0 bus and that no other device is connected at the same USB controller

	- for installing the iViewX software on another laptop, you need a license key. only one license is given when 
	  purchasing the system. thus you must "unlicence" the other computer before you can license another one (cf. iViewX
 	  handbook, p.33). 


|| If you have separate computers for recording EEG and Eyetracker data, then those have to communicate via ethernet. To do so, you have
   to specify the IP address in the matlab script in the EEG laptop (before calling smi('connect')) and in the iViewX software (Setup-->
   Hardware-->Communication, cf. iViewX handbook p.63)


|| For analyzing the eyetracking data, convert the binary data file to an .idf (ASCII) file with the utility function provided
   on the iViewX CD. The idf-files can be read into matlab with the function eye_readIDF.m
   To synchronize EEG and ET data, use the function eye_synchEEG.m
   All analysis that is specific to ET data has the prefix eye_*, but many other other bbci toolbox functions can be applied
   to ET data as well (cf. documentation of the eye_* functions).


 