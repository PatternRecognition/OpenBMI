%% erd

clc; close all; clear all;
%% file
dd='C:\Users\Doyeunlee\Desktop\Analysis\rawdata\';


% reaching
% filelist={'jmlee_reaching_MI'};
% filelist={'eslee_reaching_MI'};
filelist={'dslim_reaching_realMove'};

% multigrasp
% filelist={'eslee_multigrasp_MI','jmlee_multigrasp_MI','dslim_multigrasp_MI'};
% filelist={'eslee_multigrasp_realMove','jmlee_multigrasp_realMove','dslim_multigrasp_realMove'};

% twist
% filelist={'eslee_twist_MI','jmlee_twist_MI','dslim_twist_MI'};
% filelist={'eslee_twist_realMove','jmlee_twist_realMove','dslim_twist_realMove'};

for i = 1:length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}]);
    
    band = [4 40];
    [b,a]=butter(5, band/cnt.fs*2);
    cnt_flt=proc_channelwise(cnt, 'filtfilt',b,a);
    
    ival = [-500 3000];
    epo = cntToEpo(cnt_flt, mrk, ival);
    
    fv = proc_rectifyChannels(epo);
    fv = proc_movingAverage(fv, 200, 'centered');
    fv = proc_baseline(fv, [-500 0]);
    erd = proc_average(fv);
    
    erd_ref = proc_classMean(erd);
    erd_ref.className = {'mean'};
    erd=proc_subtractReferenceClass(erd, erd_ref);
    
    w=erd.x(351,:,1);
    length(mnt.x);
    length(w);
    plotScalpPattern(mnt, w)
    
    saveas(figure(1), 'erd_dslim_reaching_realMove.png');
end
    
    
    
    
    
    
    