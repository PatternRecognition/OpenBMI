clc; clear all; close all;

%% plot scalp topographies of ERDs
%% get converted data

dd='dir';
filelist={'subj1','subj2'};

Result=zeros(length(filelist),1);
Result_Std=zeros(length(filelist),1);


for i=1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    band=[8 25];
    %% Define channel layout
    
    grd=sprintf('C3,Cz,C4');
    mnt=mnt_setGrid(mnt,grd);
    
    %% band pass filter to the continuous EEG
    [b,a]=butter(5, band/cnt.fs*2);
    cnt_flt=proc_channelwise(cnt,'filtfilt',b,a);
    
    %% cutout segments (short-time windows) from the continuous signals
    epo=makeEpochs(cnt_flt,mrk,[-500 4000]);
    epo=proc_rectifyChannels(epo);
    epo=proc_movingAverage(epo,200,'centered');
    epo=proc_baseline(epo, [-500 0]);
    erd=proc_average(epo);
    
    scalpEvolutionPlusChannel(erd,mnt,'C4',1000:750:4000);
end
