%--------------------------------------------------------------------------
% 60EEG_4EOG_7EMG_converter.m
% This file is about convert raw data file to .m (converted) file
%
%
%--------------------------------------------------------------------------
%% initializing
clear all; clc; close all;

%% file list
% Write down where raw data file downloaded (file directory)
dd='C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\';
% Write down file name that you want to convert
filelist={'session1_sub1_multigrasp_MI'};

%%
for ff= 1:length(filelist)
    
    file= filelist{ff};
    opt= [];
    
    
    fprintf('** Processing of %s **\n', file);
    
    % load the header file
    try
        hdr= eegfile_readBVheader([dd '\' file]);
    catch
        fprintf('%s/%s not found.\n', dd, file);
        continue;
    end
    
    % filtering with Chev filter
    Wps= [42 49]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
    [filt.b, filt.a]= cheby2(n, 50, Ws);
    
    %% Load channel information
    %
    [cnt, mrk_orig]= eegfile_loadBV([dd '\' file],  ...
        'filt',filt,'fs',1000);
    
    % Setting save directory
    cnt.title= ['C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\' file];
    % Load mrk file, Assign the trigger information into mrk variable
    % If you want to convert another task's data, please check the trigger
    % information into WAM_20170814_Imag_Arrow function.
    mrk = WAM_20170814_Imag_Arrow(mrk_orig);
    
    % Assign the channel montage information into mnt variable
    mnt = getElectrodePositions(cnt.clab);
    
    % Assign the sampling rate into fs_orig variable
    fs_orig= mrk_orig.fs;
    
    var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig, 'hdr',hdr};
    
    % Convert the .eeg raw data file to .mat file
    eegfile_saveMatlab(cnt.title, cnt, mrk, mnt, ...
        'channelwise',1, ...
        'format','int16', ...
        'resolution', NaN);
end

% If this scentence show in command window, raw data converted well
disp('All EEG Data Converting was Done!');