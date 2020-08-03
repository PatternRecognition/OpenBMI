function bbci= setup_idabbci_2player_calibration(player)
%SETUP_IDABBCI_2PLAYER_CALIBRATION
%
%Synopsis:
% bbci= setup_idabbci_2player
% bbci= setup_idabbci_2player(player)
%
%Arguments:
% player: number of player for which the classifier should be trained,
%    default: 1

if nargout==0,
  error('You need to specify one output argument.');
end

if nargin==0,
  player= 1;
end

setup_bbci_bet_unstable; %% needed for acquire_bv

% First of all check that the parllelport is properly conneted.
bvr_checkparport('type', 'S');

fprintf('\n\nWelcome to IDA-BBCI 2-Player-Setup\n\n');
fprintf('BrainVision Recorder should be running otherwise start it and restart setup_idabbci_2player_calibration.\n\n')

bvr_sendcommand('loadworkspace', 'eci_2x64_ac_EMGlrf');
bvr_sendcommand('viewsignals');

global VP_CODE TODAY_DIR LOG_DIR REMOTE_RAW_DIR
VP_CODE= 'Duo';
acq_getDataFolder('log_dir', 1);

clear bbci
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'imag_lett*');
bbci.classDef= {1, 2, 3;      ...
                'left', 'right', 'foot'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.player= player;
bbci.save_name= [TODAY_DIR 'bbci_classifier_player' int2str(bbci.player)];

setup_idabbci_2player_general;
