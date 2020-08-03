% ====================
% CSD Toolbox Tutorial
% ====================
%
% ExampleCode1.m
% --------------
% MatLab code to supplement the CSD Toolbox online tutorial
% (URL: "http://psychophysiology.cpmc.columbia.edu/CSDtoolbox.htm")
%
% Updated: $Date: 2009/05/20 19:34:00 $ $Author: jk $
%

% ------------ Step 1 -----------------------------------------------------
% understand the spherical coordinate system of the CSD toolbox
% ------------ Step 2 -----------------------------------------------------
E = textread('E31.asc','%s');                                      
M = ExtractMontage('10-5-System_Mastoids_EGI129.csd',E);  
MapMontage(M);
% ------------ Step 3 -----------------------------------------------------
tic
[G,H] = GetGH(M);
toc
% ------------ Step 4 -----------------------------------------------------
D = textread('NR_C66_trr.dat');
D = D';
% ------------ Step 5 -----------------------------------------------------
tic
X = CSD (D, G, H);
toc
close(gcf);      % close MapMontage figure
X = X';
plot(X);
plot(X(:,[14 24]));
D = D';
plot(D);
plot(D(:,[14 24]));
% ------------ Step 6 -----------------------------------------------------
WriteMatrix2Text(X,'CSD_C66_trr.dat');
