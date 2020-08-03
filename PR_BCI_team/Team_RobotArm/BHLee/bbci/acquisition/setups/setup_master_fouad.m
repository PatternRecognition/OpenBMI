

% addpath([BCI_DIR 'acquisition/stimulation/master_fouad']);

% setup_bbci_online; %% needed for acquire_bv
% bvr_sendcommand('loadworkspace', 'EasyCap_motor_dense');
% 
% try,
%   bvr_checkparport('type','S');
% catch
%   error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
% end

% run('D:\svn\bbci\toolbox\startup\startup_tubbci2.m')
%  VP_CODE = 'VPjq';
% global TODAY_DIR REMOTE_RAW_DIR
% acq_getDataFolder;%('log_dir',1);
% REMOTE_RAW_DIR= TODAY_DIR;

opt=[];
N=20;

[cue_seq twitch_seq]= stim_sssepBCI(N, opt);
save(['D:\svn\bbci\acquisition\stimulation\master_fouad\cue_seq_dif02',VP_CODE],'cue_seq')
save(['D:\svn\bbci\acquisition\stimulation\master_fouad\twitch_seq_dif02',VP_CODE],'twitch_seq')

sum(twitch_seq,1)       