%% Comparison of Brain Activation during Motor Imagery and Motor Execution Using EEG signals
% PSD ME&MI

%% Initializing
clc;
close all;
clear all;

%% file
dd='G:\biosig4octmat-3.6.0.tar\run1_MIME\';
% Motor Imagery
filelist={'subject01_ME'};
% filelist={'eslee_multigrasp_MI','jmlee_multigrasp_MI','dslim_multigrasp_MI'};
% filelist={'eslee_twist_MI','jmlee_twist_MI','dslim_twist_MI'};

% Motor Execution
% filelist={'eslee_reaching_realMove','jmlee_reaching_realMove','dslim_reaching_realMove'};
% filelist={'eslee_multigrasp_realMove','jmlee_multigrasp_realMove','dslim_multigrasp_realMove'};
% filelist={'eslee_twist_realMove','jmlee_twist_realMove','dslim_twist_realMove'};

% 0~3 s
ival=[0 3001];

%% 
for i = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % IIR filter (Butterworth)
    cnt = proc_filtButter(cnt, 5, [4 40]);
    % Preprocessing, spatial filtering - CAR
    cnt = proc_commonAverageReference(cnt);
    % cnt to epoch
    epo = cntToEpo(cnt,mrk,ival);
    % Select channels
    epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
        'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
        'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
    
    % delare variables
    classes=size(epo.className,2);
    % task º° trial ¼ö
    trial=50;
    
    % extract the 'Rest' class
%     for ii=1:classes
%         if strcmp(epo.className{ii},'Rest')
%             epoRest=proc_selectClasses(epo,{epo.className{ii}});
%             epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
%             epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
%         else
%             epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
%             epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
%             epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
%         end
%     end
%     if classes<6
%         epo_check(size(epo_check,2)+1)=epoRest;
%     end
%     %concatenate the classes
%     for ii=1:size(epo_check,2)
%         if ii==1
%             concatEpo=epo_check(ii);
%         else
%             concatEpo=proc_appendEpochs(concatEpo, epo_check(ii));
%         end
%     end
    
    % PSD
    rng default
    Fs = 250;
    t = 0:1/Fs:1-1/Fs;
    x = cos(2*pi*100*t) + randn(size(t));
    
    N = length(x);
    xdft = fft(x);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N))*abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(x):Fs/2;
    
    
    
    
end
% end

disp('Done');
