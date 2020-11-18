% gigascience -> EMG signal figure

clear all; clc; close all;

dd='C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\';
filelist = {'session1_sub4_twist_realMove_EMG'};
fold=5;
ival=[0 4001];
filtBank = [50 250]; 

% selected_class = [11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53];
selected_class = [65:70];

for i = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
%     cnt = proc_commonAverageReference(cnt, selected_class);
    cnt=proc_selectChannels(cnt,selected_class);
    cnt = proc_filtButter(cnt, 5, filtBank);
    epo=cntToEpo(cnt,mrk,ival);

    classes=size(epo.className,2);
    trial=size(epo.x,3)/2/(classes-1);
    
    eachClassFold_no=trial/fold;
end


ival = [-500 4000];

epo_1 = cntToEpo(cnt, mrk, ival);
fv_1 = proc_rectifyChannels(epo_1);
fv_1 = proc_movingAverage(fv_1, 100, 'centered');
fv_1 = proc_baseline(fv_1, [-500 0]);

result = mean(fv_1.x,3); 
a = result;

X = abs(a);
X_1 = movmean(X, 30);

plot(X_1);
% plot(X_1,'LineWidth',1.5);
% ylim([0 200]);
xticks([0 1250 3750 6250 8750 11250]);
xlim([0 11251]);
ylim([0 4]);