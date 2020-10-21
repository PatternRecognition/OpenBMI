%%
clear all; clc; close all;

%% file list
dd='H:\0_RawData\';

% filelist={'session1_sub7_reaching_MI','session1_sub7_reaching_realMove','session1_sub22_reaching_MI','session1_sub22_reaching_realMove','session2_sub7_reaching_MI','session2_sub7_reaching_realMove','session2_sub17_reaching_MI','session2_sub17_reaching_realMove','session2_sub18_reaching_MI','session2_sub18_reaching_realMove','session3_sub7_reaching_MI','session3_sub7_reaching_realMove','session3_sub8_reaching_MI','session3_sub8_reaching_realMove','session3_sub11_reaching_MI','session3_sub11_reaching_realMove','session3_sub19_reaching_MI','session3_sub19_reaching_realMove','session3_sub25_reaching_MI','session3_sub25_reaching_realMove'};
% filelist={'session1_sub7_multigrasp_MI','session1_sub7_multigrasp_realMove','session1_sub22_multigrasp_MI','session1_sub22_multigrasp_realMove','session2_sub7_multigrasp_MI','session2_sub7_multigrasp_realMove','session2_sub17_multigrasp_MI','session2_sub17_multigrasp_realMove','session2_sub18_multigrasp_MI','session2_sub18_multigrasp_realMove','session3_sub7_multigrasp_MI','session3_sub7_multigrasp_realMove','session3_sub8_multigrasp_MI','session3_sub8_multigrasp_realMove','session3_sub11_multigrasp_MI','session3_sub11_multigrasp_realMove','session3_sub19_multigrasp_MI','session3_sub19_multigrasp_realMove','session3_sub25_multigrasp_MI','session3_sub25_multigrasp_realMove'};
filelist={'session1_sub7_twist_MI','session1_sub7_twist_realMove','session1_sub22_twist_MI','session1_sub22_twist_realMove','session2_sub7_twist_MI','session2_sub7_twist_realMove','session2_sub17_twist_MI','session2_sub17_twist_realMove','session2_sub18_twist_MI','session2_sub18_twist_realMove','session3_sub7_twist_MI','session3_sub7_twist_realMove','session3_sub8_twist_MI','session3_sub8_twist_realMove','session3_sub11_twist_MI','session3_sub11_twist_realMove','session3_sub19_twist_MI','session3_sub19_twist_realMove','session3_sub25_twist_MI','session3_sub25_twist_realMove'};

% % filelist={'session1_sub1_multigrasp_realMove','session1_sub1_multigrasp_MI','session1_sub2_multigrasp_realMove','session1_sub2_multigrasp_MI','session1_sub3_multigrasp_realMove','session1_sub3_multigrasp_MI','session1_sub4_multigrasp_realMove','session1_sub4_multigrasp_MI','session1_sub5_multigrasp_realMove','session1_sub5_multigrasp_MI','session1_sub6_multigrasp_realMove','session1_sub6_multigrasp_MI','session1_sub7_multigrasp_realMove','session1_sub7_multigrasp_MI','session1_sub8_multigrasp_realMove','session1_sub8_multigrasp_MI','session1_sub9_multigrasp_realMove','session1_sub9_multigrasp_MI','session1_sub10_multigrasp_realMove','session1_sub10_multigrasp_MI','session1_sub11_multigrasp_realMove','session1_sub11_multigrasp_MI','session1_sub12_multigrasp_realMove','session1_sub12_multigrasp_MI','session1_sub13_multigrasp_realMove','session1_sub13_multigrasp_MI'};
% % filelist={'session2_sub1_multigrasp_realMove','session2_sub1_multigrasp_MI','session2_sub2_multigrasp_realMove','session2_sub2_multigrasp_MI','session2_sub3_multigrasp_realMove','session2_sub3_multigrasp_MI','session2_sub4_multigrasp_realMove','session2_sub4_multigrasp_MI','session2_sub5_multigrasp_realMove','session2_sub5_multigrasp_MI','session2_sub6_multigrasp_realMove','session2_sub6_multigrasp_MI','session2_sub7_multigrasp_realMove','session2_sub7_multigrasp_MI','session2_sub8_multigrasp_realMove','session2_sub8_multigrasp_MI','session2_sub9_multigrasp_realMove','session2_sub9_multigrasp_MI','session2_sub10_multigrasp_realMove','session2_sub10_multigrasp_MI','session2_sub11_multigrasp_realMove','session2_sub11_multigrasp_MI','session2_sub12_multigrasp_realMove','session2_sub12_multigrasp_MI','session2_sub13_multigrasp_realMove','session2_sub13_multigrasp_MI'};
% % filelist={'session3_sub1_multigrasp_realMove','session3_sub1_multigrasp_MI','session3_sub2_multigrasp_realMove','session3_sub2_multigrasp_MI','session3_sub3_multigrasp_realMove','session3_sub3_multigrasp_MI','session3_sub4_multigrasp_realMove','session3_sub4_multigrasp_MI','session3_sub5_multigrasp_realMove','session3_sub5_multigrasp_MI','session3_sub6_multigrasp_realMove','session3_sub6_multigrasp_MI','session3_sub7_multigrasp_realMove','session3_sub7_multigrasp_MI','session3_sub8_multigrasp_realMove','session3_sub8_multigrasp_MI','session3_sub9_multigrasp_realMove','session3_sub9_multigrasp_MI','session3_sub10_multigrasp_realMove','session3_sub10_multigrasp_MI','session3_sub11_multigrasp_realMove','session3_sub11_multigrasp_MI','session3_sub12_multigrasp_realMove','session3_sub12_multigrasp_MI','session3_sub13_multigrasp_realMove','session3_sub13_multigrasp_MI'};
% % filelist={'session1_sub1_reaching_realMove','session1_sub1_reaching_MI','session1_sub2_reaching_realMove','session1_sub2_reaching_MI','session1_sub3_reaching_realMove','session1_sub3_reaching_MI','session1_sub4_reaching_realMove','session1_sub4_reaching_MI','session1_sub5_reaching_realMove','session1_sub5_reaching_MI','session1_sub6_reaching_realMove','session1_sub6_reaching_MI','session1_sub7_reaching_realMove','session1_sub7_reaching_MI','session1_sub8_reaching_realMove','session1_sub8_reaching_MI','session1_sub9_reaching_realMove','session1_sub9_reaching_MI','session1_sub10_reaching_realMove','session1_sub10_reaching_MI','session1_sub11_reaching_realMove','session1_sub11_reaching_MI','session1_sub12_reaching_realMove','session1_sub12_reaching_MI','session1_sub13_reaching_realMove','session1_sub13_reaching_MI'};
% % filelist={'session2_sub1_reaching_realMove','session2_sub1_reaching_MI','session2_sub2_reaching_realMove','session2_sub2_reaching_MI','session2_sub3_reaching_realMove','session2_sub3_reaching_MI','session2_sub4_reaching_realMove','session2_sub4_reaching_MI','session2_sub5_reaching_realMove','session2_sub5_reaching_MI','session2_sub6_reaching_realMove','session2_sub6_reaching_MI','session2_sub7_reaching_realMove','session2_sub7_reaching_MI','session2_sub8_reaching_realMove','session2_sub8_reaching_MI','session2_sub9_reaching_realMove','session2_sub9_reaching_MI','session2_sub10_reaching_realMove','session2_sub10_reaching_MI','session2_sub11_reaching_realMove','session2_sub11_reaching_MI','session2_sub12_reaching_realMove','session2_sub12_reaching_MI','session2_sub13_reaching_realMove','session2_sub13_reaching_MI'};
% % filelist={'session3_sub1_reaching_realMove','session3_sub1_reaching_MI','session3_sub2_reaching_realMove','session3_sub2_reaching_MI','session3_sub3_reaching_realMove','session3_sub3_reaching_MI','session3_sub4_reaching_realMove','session3_sub4_reaching_MI','session3_sub5_reaching_realMove','session3_sub5_reaching_MI','session3_sub6_reaching_realMove','session3_sub6_reaching_MI','session3_sub7_reaching_realMove','session3_sub7_reaching_MI','session3_sub8_reaching_realMove','session3_sub8_reaching_MI','session3_sub9_reaching_realMove','session3_sub9_reaching_MI','session3_sub10_reaching_realMove','session3_sub10_reaching_MI','session3_sub11_reaching_realMove','session3_sub11_reaching_MI','session3_sub12_reaching_realMove','session3_sub12_reaching_MI','session3_sub13_reaching_realMove','session3_sub13_reaching_MI'};
% % filelist={'session1_sub1_twist_realMove','session1_sub1_twist_MI','session1_sub2_twist_realMove','session1_sub2_twist_MI','session1_sub3_twist_realMove','session1_sub3_twist_MI','session1_sub4_twist_realMove','session1_sub4_twist_MI','session1_sub5_twist_realMove','session1_sub5_twist_MI','session1_sub6_twist_realMove','session1_sub6_twist_MI','session1_sub7_twist_realMove','session1_sub7_twist_MI','session1_sub8_twist_realMove','session1_sub8_twist_MI','session1_sub9_twist_realMove','session1_sub9_twist_MI','session1_sub10_twist_realMove','session1_sub10_twist_MI','session1_sub11_twist_realMove','session1_sub11_twist_MI','session1_sub12_twist_realMove','session1_sub12_twist_MI','session1_sub13_twist_realMove','session1_sub13_twist_MI'};
% % 실험컴filelist={'session2_sub1_twist_realMove','session2_sub1_twist_MI','session2_sub2_twist_realMove','session2_sub2_twist_MI','session2_sub3_twist_realMove','session2_sub3_twist_MI','session2_sub4_twist_realMove','session2_sub4_twist_MI','session2_sub5_twist_realMove','session2_sub5_twist_MI','session2_sub6_twist_realMove','session2_sub6_twist_MI','session2_sub7_twist_realMove','session2_sub7_twist_MI','session2_sub8_twist_realMove','session2_sub8_twist_MI','session2_sub9_twist_realMove','session2_sub9_twist_MI','session2_sub10_twist_realMove','session2_sub10_twist_MI','session2_sub11_twist_realMove','session2_sub11_twist_MI','session2_sub12_twist_realMove','session2_sub12_twist_MI','session2_sub13_twist_realMove','session2_sub13_twist_MI'};
% % 실험컴filelist={'session3_sub1_twist_realMove','session3_sub1_twist_MI','session3_sub2_twist_realMove','session3_sub2_twist_MI','session3_sub3_twist_realMove','session3_sub3_twist_MI','session3_sub4_twist_realMove','session3_sub4_twist_MI','session3_sub5_twist_realMove','session3_sub5_twist_MI','session3_sub6_twist_realMove','session3_sub6_twist_MI','session3_sub7_twist_realMove','session3_sub7_twist_MI','session3_sub8_twist_realMove','session3_sub8_twist_MI','session3_sub9_twist_realMove','session3_sub9_twist_MI','session3_sub10_twist_realMove','session3_sub10_twist_MI','session3_sub11_twist_realMove','session3_sub11_twist_MI','session3_sub12_twist_realMove','session3_sub12_twist_MI','session3_sub13_twist_realMove','session3_sub13_twist_MI'};
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
    %% 100Hzfh
    % Load channel information
    % EEG
    [cnt, mrk_orig]= eegfile_loadBV([dd '\' file],  ...
        'filt',filt,'clab',{'not','EMG*','hEOG_L','hEOG_R','vEOG_U','vEOG_D'},'fs',250);
    % EMG
%     [cnt, mrk_orig]= eegfile_loadBV([dd '\' file],  ...
%         'filt',filt,'clab',{'not','hEOG_L','hEOG_R','vEOG_U','vEOG_D', 'Fp1','AF7','AF3','AFz','F7','F5','F3','F1','Fz','FT7','FC5','FC3','FC1','T7','C5','C3','C1','Cz','TP7','CP5','CP3','CP1','CPz','P7','P5','P3','P1','Pz','PO7','PO3','POz','Fp2','AF4','AF8','F2','F4','F6','F8','FC2','FC4','FC6','FT8','C2','C4','C6','T8','CP2','CP4','CP6','TP8','P2','P4','P6','P8','PO4','PO8','O1','Oz','O2','Iz'},'fs',2500);
    % EOG
%     [cnt, mrk_orig]= eegfile_loadBV([dd '\' file],  ...
%         'filt',filt,'clab',{'not','EMG*','Fp1','AF7','AF3','AFz','F7','F5','F3','F1','Fz','FT7','FC5','FC3','FC1','T7','C5','C3','C1','Cz','TP7','CP5','CP3','CP1','CPz','P7','P5','P3','P1','Pz','PO7','PO3','POz','Fp2','AF4','AF8','F2','F4','F6','F8','FC2','FC4','FC6','FT8','C2','C4','C6','T8','CP2','CP4','CP6','TP8','P2','P4','P6','P8','PO4','PO8','O1','Oz','O2','Iz'},'fs',2500);
    
    cnt.title= ['H:\0_ConvertedData\' file];
%         cnt.title= ['F:\converted\EMG\' 'EMG_' file];
%         cnt.title= ['F:\converted\EOG\' 'EOG_' file];
    
    
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

disp('All EEG Data Converting was Done!');
