%% Load MI data
clc; clear all; close all;
startup_bbci_toolbox

BTB.DataDir = 'E:\ear_data\exp3m';

% ERP
BTB.subNum = [1 3 4 5 6 9 10];

band= [0.1 50];
t_remain = 2000; % mili seconds

%% changing setting
BTB.paradigm = 'MI';
BTB.session = 1;
BTB.modal = 'cap';
filename = sprintf('SM%02d_%d_%s',BTB.subNum(1),BTB.session,BTB.modal); 

% BTB.folderName = 's0_bhlee';
% subNum = 8;
% 6번 따로 처리해 줘야함. 4번 IMU_vis_ERP_0 데이터 이상

%% setting
BTB.MatDir= [BTB.DataDir '\' BTB.paradigm '\' 'Mat'];
BTB.RawDir = [BTB.DataDir '\' BTB.paradigm];

switch(BTB.paradigm)
    case 'SSVEP'
        % Define some settings SSVEP
        disp_ival= [0 4000]; % SSVEP
        trig_all = {1,2,3, 11, 22; '5.45','8.57','12','Start','End'};
        trig_sti = {1,2,3; '5.45','8.57','12'};
    case {'vis_ERP', 'aud_ERP','vis_oddball','ERP','ERP_bp'}
        disp_ival= [-200 800]; % ERP
        ref_ival= [-200 0] ;
        trig_all = {2,1, 11, 22; 'target','non-target','Start','End'};
        trig_sti = {2,1 ;'target','non-target'};
    case 'MI'
        disp_ival = [0 5000];
        trig_all = {1,2,3,11,22; 'right','left','rest','Start','End'};
        trig_sti = {1,2,3; 'right','left','rest'};
end

%% Cap
sub = 1;
sess = BTB.session;
% load data
% filename = [BTB.subject '_' BTB.paradigm '_' 'tr' '_cap'];

[cnt, mrk_orig, hdr] = file_readBV(filename, 'Fs', 500);

% % Apply bandpass filter to reduce drifts
[b,a]= butter(5, band/cnt.fs*2);
cap_cnt{sub,sess}= proc_filtfilt(cnt, b, a);

% create mrk
cap_mrk{sub,sess}= mrk_defineClasses(mrk_orig, trig_all);
cap_mrk{sub,sess}.orig= mrk_orig;
cap_mrk{sub,sess} = mrk_defineClasses(cap_mrk{sub,sess}, trig_sti);


% segmentation
cap_epo{sub,sess} = proc_segmentation(cap_cnt{sub,sess}, cap_mrk{sub,sess}, disp_ival);

% create mnt
cap_mnt= mnt_setElectrodePositions(cap_epo{sub,sess}.clab);

% save in matlab format

% b = load([BTB.MatDir '\s4.mat'])
% cap_cnt2(2,:) = a.cap_cnt
% 
% [cnt2, mrk2, mnt2] = file_loadMatlab('s1');
%% Ear 
% load data
[cnt, mrk_orig, hdr] = file_readBV(filename, 'Fs', 500);

% % Apply highpass filter to reduce drifts
[b,a]= butter(5, band/cnt.fs*2);
cnt= proc_filtfilt(cnt, b, a);

% create mrk
ear_mrk{sub,sess}= mrk_defineClasses(mrk_orig, trig_all);
ear_mrk{sub,sess}.orig= mrk_orig;
ear_mrk{sub,sess} = mrk_defineClasses(ear_mrk{sub,sess}, trig_sti);

% preprocessing
cnt = proc_selectChannels(cnt, 1:18);
% ear_i_cnt{sub,sess} = proc_selectChannels(cnt, isolated_ear_chan);
% ear_cnt{sub,sess} = proc_removeChannels(cnt, isolated_ear_chan);
ear_cnt{sub,sess} = cnt;

% segmentation
ear_epo{sub,sess} = proc_segmentation(ear_cnt{sub,sess}, ear_mrk{sub,sess}, disp_ival);
% ear_i_epo{sub,sess} = proc_segmentation(ear_i_cnt{sub,sess}, ear_seg_mrk{sub,sess}, disp_ival);

% offset_iear{j}(i,:) = mean(ear_i_cnt{sub,sess}.x);

% create mnt
ear_mnt= mnt_setElectrodePositions(cnt.clab);

% isolated_chan{1,1} = isolated_cap_chan;
% isolated_chan{1,2} = isolated_ear_chan;

disp(STOP)
%% End
% save([BTB.MatDir '\' BTB.folderName '_' BTB.subject], 'BTB', 'cap_cnt', 'cap_seg_mrk', 'cap_mnt', 'ear_cnt','ear_seg_mrk', 'ear_mnt','IMU_cnt', 'IMU_seg_mrk');
% 
% disp('Saved data')

