% amplitude spectra
% imagery and movement

clc; close all; clear all;

dd='C:\Users\Doyeunlee\Desktop\Journal_dylee\1_ConvertedData\';
fsldx=2;
fs={'100','250','1000'};
ref='FCz';
channelMatrix={'F3','F1','Fz','F2','F4';
    'FC3','FC1','FCz','FC2','FC4';
    'C3','C1', 'Cz', 'C2', 'C4';
    'CP3','CP1','CPz','CP2','CP4'
    'P3','P1','Pz','P2','P4'};

% Motor Imagery
% filelist={'sub01','sub02','sub03','sub04','sub05','sub06','sub07','sub08','sub09','sub10'};
filelist={'sub01'};

for sub = 1:length(filelist)
    ival = [0 3000];
    % Data load - MI
    [cntReach,mrkReach,mntReach]=eegfile_loadMatlab([dd 'MI' '\' fs{fsldx} '\' filelist{sub} '_reaching_' 'MI']);
    % Data load - realMove
    [cntReach_ME,mrkReach_ME,mntReach_ME]=eegfile_loadMatlab([dd 'realMove' '\' fs{fsldx} '\' filelist{sub} '_reaching_' 'realMove']);
    
    % butter
    cntReach=proc_filtButter(cntReach,5,[4 40]);
    cntReach_ME=proc_filtButter(cntReach_ME,5,[4 40]);
    
    epoReach=cntToEpo(cntReach,mrkReach,ival);
    epoReach_ME=cntToEpo(cntReach_ME,mrkReach_ME,ival);
    
    epoReach=proc_selectChannels(epoReach,{'F3','F1','Fz','F2','F4',...
        'FC3','FC1','FCz','FC2','FC4',...
        'C3','C1', 'Cz', 'C2', 'C4', ...
        'CP3','CP1','CPz','CP2','CP4',...
        'P3','P1','Pz','P2','P4'});
    
    epoReach_ME=proc_selectChannels(epoReach_ME,{'F3','F1','Fz','F2','F4',...
        'FC3','FC1','FCz','FC2','FC4',...
        'C3','C1', 'Cz', 'C2', 'C4', ...
        'CP3','CP1','CPz','CP2','CP4',...
        'P3','P1','Pz','P2','P4'});
    
    fs = 250;
    
    %
    plot(psd(spectrum.periodogram, cntReach_ME.x, 'Fs', fs));
    psdest = psd(spectrum.periodogram, cntReach_ME.x, 'Fs', fs);
    plot(psdest.Frequencies,psdest.Data);
    xlabel('Hz'); grid on;
    hold on;
    psdest = psd(spectrum.periodogram, cntReach.x, 'Fs', fs);
    plot(psdest.Frequencies,psdest.Data);
    xlabel('Hz'); grid on;
    
    % 
    xdft = fft(cntReach_ME.x);
    xdft = xdft(1:length(cntReach_ME.x)/2+1);
    freq = 0:fs/length(cntReach_ME.x):fs/2;
    figure();
    plot(freq,abs(xdft));
    xlabel('Hz');
    
    %
    y= fft(cntReach.x);
    n = length(cntReach.x);
    f = (0:n-1)*(fs/n);
    power = abs(y).^2/n;
    
    plot(f,power);
    xlabel('Frequency');
    ylabel('Power');
    
    y= fft(cntReach_ME.x);
    n = length(cntReach_ME.x);
    f = (0:n-1)*(fs/n);
    power = abs(y).^2/n;
    
    plot(f,power);
    xlabel('Frequency');
    ylabel('Power');
    
    
    
    
    
end







