setup_bbci_bet_unstable; %% needed for acquire_bv

% First of all check that the parllelport is properly conneted.
%bvr_checkparport;
%bvr_checkparport('type', 'R');

global VP_CODE TODAY_DIR LOG_DIR
VP_CODE= 'Duo';
acq_getDataFolder('log_dir', 1);

setup_idabbci_2player_general;
fprintf('Remember to use\nplayer=2; matlab_feedbacks\nfor player 2.\n');
