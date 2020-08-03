clc;clear all;close all;

 file = 'D:\data\bbciRaw\VPiz_08_04_15\fingertip_rightVPiz';
 [cnt,mrk]= eegfile_loadBV(file,'fs',100);

[b,a]= butter(5, [4 6]/cnt.fs*2);
cnt_filt = proc_filtfilt(cnt,b,a)


ind = chanind(cnt_filt,'C3');
data_cnt = cnt_filt.x(:,ind);

figure
t = [0:1/cnt.fs:length(data_cnt)/cnt.fs-(1/cnt.fs)];
plot(t,data_cnt)


data_clab = cnt.clab(ind);
data_cnt_spec = abs(fft(data_cnt));

f =((0:length(data_cnt)-1)/length(data_cnt))*cnt.fs;  

figure
plot(f,data_cnt_spec)

title(['Spectrum of channel ', char(data_clab)])