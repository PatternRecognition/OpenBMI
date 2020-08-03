close all;clear all;clc
 file = 'D:\data\bbciRaw\VPiz_08_04_15\fingertip_rightVPiz';
 [cnt,mrk]= eegfile_loadBV(file,'fs',100);

 Para.fs = 22050;
Para.act_time = 2;
Para.ref_time = 5;
Para.modfreq = [5:2:31];
Para.carfreq = 200;
Para.num_trial = 20;
Para.count_dura = 10;
Para.ifi = 0.5;
Para.num_block = 2;
 
 
 length_trial = Para.ref_time+(Para.act_time+Para.ifi)*length(Para.modfreq);
t_trial = [0:1/cnt.fs:length_trial-(1/cnt.fs)];
 
 
 
ind = chanind(cnt,'C3');
data_cnt = cnt.x(:,ind);
figure
t = [0:1/cnt.fs:length(data_cnt)/cnt.fs-(1/cnt.fs)];
plot(t,data_cnt)
data_clab = cnt.clab(ind);
data_cnt_spec = abs(fft(data_cnt));
f =((0:length(data_cnt)-1)/length(data_cnt))*cnt.fs;  
figure
plot(f,data_cnt_spec)
xlim([0 40])
title(char(data_clab))

ny_freq = cnt.fs/2;
  m = [0 0 1 1 0 0];
  ff = [0 1/ny_freq 2/ny_freq 39/ny_freq 40/ny_freq 1];
 a = fir2(8000,ff,m);
 b=1;
data_cnt_filt = filter(a,b,data_cnt);
data_cnt_filt_spec = abs(fft(data_cnt_filt));

figure 
plot(f,data_cnt_filt_spec)
 xlim([0 ny_freq])
title(['Prefilteret ',char(data_clab)])

figure
mrk_num.S102 = 102;
mrk_num.S103 = 103;
mrk_num.Stim = [5:2:31];

mrk_pos = get_mrkPos(mrk,mrk_num);
trial = cntToEpoch(data_cnt,mrk_pos.S102,40,cnt.fs);

plot_Trial(trial(:,10),t_trial,mrk_pos,cnt.fs,[5:2:31])