% path([BCI_DIR 'acquisition/setups/vitalbci_season1'], path);
% startup_bbcilaptop;
setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to Photobrowser\n\n');

%If you like to crash the whole computer, do this:
%system('c:\Vision\Recorder\Recorder.exe &')

%If matlab crashed before, BVR might still be in recording mode
% bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
% if strcmp(general_port_fields.bvmachine, 'bbcipc'),
%   bvr_sendcommand('loadworkspace', 'FastnEasy128_EOGhv_EMGlrf');  %% Tuebingen
% else
% %   bvr_sendcommand('loadworkspace', 'FastnEasy_EMGlrf');      %% Berlin's Fast'n'Easy Caps
%   %bvr_sendcommand('loadworkspace', 'EasyCap_128_EMGlrf_EOG');     %% Berlin EasyCap
%   %bvr_sendcommand('loadworkspace', 'eci_128ch_EMGlrf');     %% Berlin Fast'n'Easy Cap (obsolete)
%   bvr_sendcommand('loadworkspace', 'one_channel');
% end
% try
%   bvr_checkparport('type','S');
% catch
%   error('BrainVision Recorder must be running.\nThen restart %s.', mfilename);
% end

global TODAY_DIR REMOTE_RAW_DIR VP_CODE DATA_DIR EEG_RAW_DIR
% DATA_DIR = '/home/martijn/data/';
% TODAY_DIR = [DATA_DIR 'bbciRaw/VPnh_10_01_27/'];
REMOTE_RAW_DIR= TODAY_DIR;
EEG_RAW_DIR = [DATA_DIR 'bbciRaw/'];
acq_makeDataFolder('log_dir',1);

[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci= [];
bbci.setup= 'PhotoBrowser';
bbci.train_file= strcat(subdir, '/P300PhotoBrowser_train_',VP_CODE, '*');
% bbci.train_file= strcat(subdir, '\OnlineTrainShortToneFile',VP_CODE, '*');
% bbci.clab= {'FC3-4', 'F5-6', 'PCP5-6', 'C5-6','CP5-6','P5-6', 'P9,7,8,10','PO7,8', 'E*'};
bbci.clab = {'*'};
bbci.func_mrk = 'mrkodef_photobrowser_oddball';
% bbci.classes= 'auto';
bbci.feedback= '1d_AEP';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
% bbci.setup_opts.usedPat= 'auto';
%If'auto' mode does not work robustly:
%bbci.setup_opts.usedPat= [1:6];
bbci.fs = 100;
bbci.fb_machine = general_port_fields(1).bvmachine; 
bbci.fb_port = 12345;

bbci.filt.b = [];bbci.filt.a = [];
Wps= [40 49]/1000*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
[bbci.filt.b, bbci.filt.a]= cheby2(n, 50, Ws);

bbci.withclassification = 1;
bbci.withgraphics = 1;