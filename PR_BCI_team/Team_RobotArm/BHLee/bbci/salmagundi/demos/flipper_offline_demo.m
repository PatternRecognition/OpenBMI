%% Matlab 1
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'simulated_amplifier');
bvr_sendcommand('viewsignals');

SUB_DIR= 'VPsab_09_03_19';
bbci_setup= strcat(EEG_RAW_DIR, SUB_DIR, '/bbci_classifier_cspauto_48chans_setup_001');

%% Start Matlab GUI in Matlab 2
system(['matlab -r "matlab_control_gui(''flipper_hardware'', ''classifier'',''' bbci_setup ''');" &']);

%% Start Online Classifier in Matlab 1
bbci_bet_apply(bbci_setup, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);
