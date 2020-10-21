%% comparison of ME and MI
% class

clc; close all; clear all;

dd = 'C:\Users\Doyeunlee\Desktop\Analysis\convert\';
filelist = {'dslim_reaching_MI'};

for i = length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}]);
    
    ival_erd = [-500 3000];
    
    band_erd= [4 40];
    ival_scalps= -800:200:200;
    [b,a]= butter(5, band_erd/cnt.fs*2);
    cnt= proc_filt(cnt, b, a);
    epo= cntToEpo(cnt, mrk, ival_erd);
    
    fv = proc_rectifyChannels(epo);
    fv = proc_movingAverage(fv, 200, 'centered');
    fv = proc_baseline(fv, [-500 0]);
    erp = proc_average(fv);
    
    clab = epo.clab;
    
    figure();
    scalpEvolutionPlusChannel(epo, mnt, clab, ival_scalps);
    hold on;

end