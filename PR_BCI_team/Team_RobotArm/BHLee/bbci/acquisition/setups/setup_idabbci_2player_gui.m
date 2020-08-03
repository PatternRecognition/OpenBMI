global VP_CODE REMOTE_RAW_DIR TODAY_DIR
setup_bbci_bet_unstable
VP_CODE= 'Duo';
acq_getDataFolder('log_dir', 1);
REMOTE_RAW_DIR= TODAY_DIR;
%% The follwing line is need if bbci_bet_apply runs on bot00
%REMOTE_RAW_DIR= '/home/neuro/data/BCI/bbciRaw/'

setup_idabbci_2player_general;
