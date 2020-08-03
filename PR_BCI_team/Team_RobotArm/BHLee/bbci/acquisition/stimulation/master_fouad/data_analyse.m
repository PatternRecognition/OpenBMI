clc;

file= 'C:\Documents and Settings\Min Konto\Dokumenter\Fouads Mappe\DTU\tiendesem\BCI-SSSEP Projekt\Matlab\eeg_data\test_master_fouad';

mrk= eegfile_readBVmarkers(file);

[cnt,mrk]= eegfile_loadBV(file,'clab',[1:20]);
Fp1 = cnt.x(:,1);
mrko = mrk.desc;
num_freq  = [17:2:31];
%%
for iii = 1:length(num_freq)
mrko_i = regexp(mrko, ['S ',int2str(num_freq(iii))]);

mrko_ii = cellfun('isempty',mrko_i);


ii = find(mrko_ii==0);
ind(iii,:) = ii;
mrk_pos(iii,:) = mrk.pos(ind(iii,:));
end

num_trial = [102:103];

for k = 1:length(num_trial)
mrko_k = regexp(mrko, ['S',int2str(num_trial(k))]);

mrko_kk = cellfun('isempty',mrko_k);

kk = find(mrko_kk==0);
ind_k(k,:) = kk;
mrk_pos_k(k,:) = mrk.pos(ind_k(k,:));
end

%%
% for ii=1:length(mrk_pos)
% 
% Fp1_trial(ii,:) = Fp1(mrk_pos(ii):(mrk_pos(ii)+25000)-1);
% 
% end
% t_trial1 = [trial1/1000:1/1000:((trial1+25000)/1000)-(1/1000)];
% 
% figure
% plot(t_trial1,Fp1_trial1)
% xlabel('Time [s]')
% ylabel('Amplitude [mV]')
% grid on
% 
% text(mrk_pos(1)/1000,32,' \bf \leftarrow S 19','Rotation',90,'FontSize',8,'VerticalAlignment','middle')
% 
% %% 
% t = [0:1/cnt.fs:length(cnt.x(:,1))/cnt.fs-(1/cnt.fs)];
% figure
% plot(t,cnt.x(:,1))
% text(mrk_pos_S105(1)/1000,32,' \bf \leftarrow S105','Rotation',90,'FontSize',8,'VerticalAlignment','middle')