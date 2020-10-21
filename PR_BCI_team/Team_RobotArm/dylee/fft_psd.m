%% Comparison of Brain Activation during Motor Imagery and Motor Execution Using EEG signals
% FFT -> PSD (band)

clc; close all; clear all;
%% file
dd='C:\Users\Doyeunlee\Desktop\Analysis\rawdata\convert\';
% reaching
% filelist={'eslee_reaching_MI','jmlee_reaching_MI','dslim_reaching_MI','eslee_reaching_realMove','jmlee_reaching_realMove','dslim_reaching_realMove'};
% filelist='dslim_reaching_MI';
% filelist='eslee_reaching_MI';
% filelist='jmlee_reaching_MI';
% filelist='dslim_reaching_realMove';
filelist='eslee_reaching_realMove';
% filelist='jmlee_reaching_realMove';

% multigrasp
% filelist={'eslee_multigrasp_MI','jmlee_multigrasp_MI','dslim_multigrasp_MI','eslee_multigrasp_realMove','jmlee_multigrasp_realMove','dslim_multigrasp_realMove'};

% twist
% filelist={'eslee_twist_MI','jmlee_twist_MI','dslim_twist_MI','eslee_twist_realMove','jmlee_twist_realMove','dslim_twist_realMove'};

[cnt,mkr,mnt]=eegfile_loadMatlab([dd filelist]);
cnt = proc_filtButter(cnt, 5, [4 40]);
cnt = proc_commonAverageReference(cnt);
% cnt.clab = cnt.clab([11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53]);
cnt.x = cnt.x( :, [11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53]);
figure(1); 
xlabel('sample'); 
ylabel('magnitude'); 
plot(cnt.x);
legend('14 channel plot of subject1 BASELINE'); 
data= cnt.x'; 
value1=data(1,:); 
chan1= value1-mean(value1); 
fs = 250;
d=1/fs; 
t=[0:length(chan1)-1]*d; 
figure(2); 
plot(chan1);
title('original signal'); 
fs=fft(chan1,128); 
pp=fs.*conj(fs)/128; 
ff=(1:64)/128/d; 
figure(3); 
plot(ff,pp(1:64)); 
xlim([4 25])
ylim([0 250])
ylabel('power spectrum density');
xlabel('frequency');
title('signal power spectrum');
saveas(figure(3), 'psd_eslee_reaching_realMove.png');


