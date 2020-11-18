% gigascience -> EMG signal figure

clear all; clc; close all;

dd='C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\';
% filelist={'sub01','sub02','sub03','sub04','sub05','sub06','sub07','sub08','sub09','sub10'};
filelist={'session1_sub4_multigrasp_MI_EMG'};
ival=[0 4500];

% selected_class = [11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53];
% selected_class = {'F3','F1','Fz','F2','F4','FC3','FC1','FCz','FC2','FC4','C3','C1', 'Cz', 'C2', 'C4','CP3','CP1','CPz','CP2','CP4','P3','P1','Pz','P2','P4'};
% selected_class = {'C3','C1', 'Cz', 'C2', 'C4'};
% selected_class = {'Cz'};
selected_class ={'EMG_1','EMG_2','EMG_3','EMG_4','EMG_5','EMG_6','EMG_ref'};
% selected_class = [1 2 3 4 5 6 7];

fold = 5;

for sub = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd '\' filelist{sub}]);
    
    cnt=proc_selectChannels(cnt,selected_class);
    cnt=proc_filtButter(cnt,5,[10 500]);
    epo=cntToEpo(cnt,mrk,ival);
    
    classes=size(epo.className,2);
    trial=size(epo.x,3)/2/(classes-1);
    
    eachClassFold_no=trial/fold;

end

fs = 250;
% plot(psd(spectrum.periodogram, cntReach.x, 'Fs', fs));
% psdest = psd(spectrum.periodogram, cntReach.x, 'Fs',fs);
% plot(psdest.Frequencies,psdest.Data);
% xlabel('Hz'); grid on;
% ylim([0 60]);
% xlim([0 60]);
% hold on;
% 
% fs = 250;
% figure();
% plot(psd(spectrum.periodogram, cntReach_ME.x, 'Fs', fs));
% psdest = psd(spectrum.periodogram, cntReach_ME.x, 'Fs',fs);
% plot(psdest.Frequencies,psdest.Data);
% xlabel('Hz'); grid on;
% ylim([0 60]);
% xlim([0 60]);
% 
% xdft = fft(cntReach_ME.x);
% xdft = xdft(1:length(cntReach_ME.x)/2+1);
% freq = 0:fs/length(cntReach_ME.x):fs/2;
% figure();
% plot(freq,abs(xdft));
% xlabel('Hz');

% T = 1/fs; 
% tx = [0:length(cnt.x)-1]/fs;
% figure();
% subplot(211);
% plot(tx,cnt.x); xlabel('Time(s)'); ylabel('Amplitude(uV)');


% mean_EEGsig = mean(cnt.x);
% max_value = max(cnt.x);
% mean_value = mean(cnt.x);
% threshold=(max_value - mean_value)/2;
% subplot(212), plot(tx, mean_value);
% xlabel('Time (s)'); ylabel('Amplitude(uV)');

% T = 1/fs; 
% tx = [0:length(cntReach.x)-1]/fs;
% figure();
% % subplot(211);
% plot(tx,cntReach.x); xlabel('Time(s)'); ylabel('Amplitude(uV)');

epo_1 = cntToEpo(cnt, mrk, ival);
fv_1 = proc_rectifyChannels(epo_1);
fv_1 = proc_movingAverage(fv_1, 100, 'centered');
fv_1 = proc_baseline(fv_1, [0 500]);

result = mean(fv_1.x,3);
a = result;

X = abs(a);
X_1 = movmean(X, 30);

plot(X_1);
% plot(X_1,'LineWidth',1.5);
% ylim([0 200]);
xticks([0 1250 3750 6250 8750 11250]);
xlim([0 11251]);
% ylim([0 35]); % reaching
ylim([0 35]); % grasp
% ylim([0 10]); % twist

