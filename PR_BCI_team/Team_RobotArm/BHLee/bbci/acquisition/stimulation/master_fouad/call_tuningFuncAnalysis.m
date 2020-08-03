% 
% 
% file = 'C:\eeg_data\VPiz_08_05_02\palm_right_fc256_level_6_freqzVPiz';
% file1 = 'C:\eeg_data\VPiz_08_05_02\palm_right_fc256_level_6_freqzVPiz02';
%  
% file_1 = 'D:\data\bbciRaw\VPiz_08_05_07\palm_right_test_256_longVPiz';
% clear all;close all;
file_2 = 'D:\data\bbciRaw\VPiz_08_06_12\test_finger_left_lev8_newVPiz';

%Para.act_time = 1;

plot_chan = {'CP3','CP4'};
plot_classes = [27];
fft_band = [7 35];
buffer= 0;
filt_on=0;

stop = 2000;
start =0;


stop_ref = 4000;
start_ref=2000;

% % 
%  wo = 50/(1000/2);  bw = wo/60;
%  [filt.b,filt.a] = iirnotch(wo,bw,-10);  
%  Wn = [10 30]/500;
% [filt.b,filt.a] = cheby2(5,60,Wn);
% filt.b=Num;
%  filt.a =ones(1,length(filt.b));
%     [filt.b,filt.a]= butter(5, [5 40]/(1000/2));
[cnt,mrk]= eegfile_loadBV(file_2,'clab',plot_chan,'fs',1000);
 %[mrk_clean, RCLAB, RTRIALS]= reject_varEventsAndChannels(cnt, mrk, [start stop], 'visualize',1);

tuningFuncAnalysis(cnt,mrk,plot_chan,plot_classes,fft_band,file_2,buffer,filt_on,stop_ref,start_ref,start,stop)