clc;clear all;close all;

% file= 'C:\Documents and Settings\Min Konto\Dokumenter\Fouads Mappe\DTU\tiendesem\BCI-SSSEP Projekt\Matlab\eeg_data\test_master_fouad_140408';
file = 'D:\data\bbciRaw\test_master_fouad_140408';
mrk= eegfile_readBVmarkers(file);

[cnt,mrk]= eegfile_loadBV(file);
mrk_num.S102 = 102;
mrk_num.S103 = 103;
mrk_num.Stim = [17:2:31];

mrk_pos = get_mrkPos(mrk,mrk_num);
[t_trial trial] = cntToEpoch(cnt,mrk_pos.S102,25,1000);

plot_Trial(trial,t_trial,mrk_pos,1000,[17:2:31])
