clc; clear all; close all;

dd = 'C:\Users\Doyeunlee\Desktop\Analysis\rawdata\convert\';
filelist = 'dslim_multigrasp_MI';

%% Band topography
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); 

% band power
band= [4 8];
ival=[0 45000];
[b,a]= butter(5, band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);

epo = cntToEpo(cnt_flt, mrk, ival);

spec= proc_spectrum(epo, [4 8]);

% H = scalpPlot(mnt, spec);
figure('Name', 'CSP Patterns'); 
plotCSPatterns(csp_fv, mnt, csp_w, csp_fv.y)

plotCSPatterns(fv, mnt, W, la);


