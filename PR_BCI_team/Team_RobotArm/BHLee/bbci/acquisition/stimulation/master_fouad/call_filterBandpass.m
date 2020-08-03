clc;close all
%file= 'C:\Documents and Settings\Min Konto\Dokumenter\Fouads Mappe\DTU\tiendesem\BCI-SSSEP Projekt\Matlab\eeg_data\test_master_fouad_140408';
file = 'D:\data\bbciRaw\VPiz_08_04_15\fingertip_rightVPiz';

%%% Initialisering %%%
Para.fs = 22050;
Para.act_time = 2;
Para.ref_time = 5;
Para.modfreq = [5:2:31];
Para.carfreq = 200;
Para.num_trial = 20;
Para.count_dura = 10;
Para.ifi = 0.5;
Para.num_block = 2;

mrk_num.S102 = 102;
mrk_num.S103 = 103;
mrk_num.Stim = [5:2:31];

chan = 'C#';

[cnt,mrk]= eegfile_loadBV(file,'fs',100);

[trial_spec t_trial data_ct_filt trial mean_trial data_set_meantrial data_set_meantrialSpec]  = filterBandpass(cnt,mrk,mrk_num,chan,Para);

% for k=1:size(meantrial,2)
%   MEANtrial(:,k) =  abs(fft(meantrial(:,k)));
%   
% 
% end
% 
% 
% %%% Plot %%%
% length_Y = size(meantrialfilt{2,1},1);
% size_Y = size(meantrialfilt,2);
% 
% f =((0:length_Y-1)/length_Y)*cnt.fs;  
% 
% for kk = 1:size_Y
%     figure('Name',['Bandpower:', char(meantrialfilt{1,kk})])
%     
%     Y = meantrialfilt{2,kk};
%     for kkk=1:size(Y,2)
%     subplot(size(Y,2)/2,2,kkk)
%     plot(f,Y(:,kkk))
%     title(['Centerfreq.', int2str(Para.modfreq(kkk))],'FontSize',12)
%     xlim([0 50])
%     ylim([0 max(max(Y))])
%     grid on
%     end
% xlabel('Frequency [Hz]')
% 
% 
% end
%    figure
%    for o=1:size_Y
% subplot(size_Y,1,o)
%     plot(f,MEANtrial(:,o))
%     title(char(meantrialfilt{1,o}))
%    end
% 
%     figure
%     for p=1:size_Y
%     subplot(size_Y,1,p)
%     plot(t_trial,meantrial(:,p))
%      title(char(meantrialfilt{1,p}))
%     end