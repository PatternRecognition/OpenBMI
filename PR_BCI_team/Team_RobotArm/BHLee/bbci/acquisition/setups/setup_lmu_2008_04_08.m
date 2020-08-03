path([BCI_DIR 'acquisition/setups/lmu_2008_04_08'], path);
path([BCI_DIR 'acquisition/stimulation/lmu_2008_04_08'], path);
setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to the LMU Experiment on 8th Apr 2008\n\n');

% Load Workspace into the BrainVision Recorder and test parallel port
bvr_sendcommand('loadworkspace', 'Aschober_Markus_keh');
try,
  bvr_checkparport('type','S');
catch
  error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
end
bvr_sendcommand('switchoffimp');

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));

global TODAY_DIR VP_CODE REMOTE_RAW_DIR
VP_CODE= 'VPhd';
acq_getDataFolder('log_dir',1);

clear bbci
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'real_handcues', VP_CODE, '*');
bbci.classDef= {1,      2;      ...
                'right II', 'right V'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.save_name= [TODAY_DIR 'classifier'];
bbci.player= 1;

fprintf('remember to run ''bbci_bet_prepare2'' after bbci_bet_prepare.\n');
