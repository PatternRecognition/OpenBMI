
VP_CODE = 'VPmaf'

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/mundus_calibration_support'], path);
path([BCI_DIR 'online/nogui'], path);

%% Settings for online classification
general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';

fprintf('\n\nWelcome to BBCI XXXX\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', ['ActiCap_64ch_motor_mundus']);
try
    bvr_checkparport('type','S');
catch
    error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
TODAY_DIR = 'D:\data\bbciRaw\VPmaf_11_08_11\';
%acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];

%% prepare settings for classifier training
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
%bbci.train_file= strcat(bbci.subdir, '/imag_arrow*');
%bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','PO*','O*'};
bbci.nPat = 5;
bbci.classes= 'auto';
%bbci.classDef= {1, 2, 3; 'left', 'right', 'foot'};
%bbci.feedback= '1d';
%bbci.setup_opts.ilen_apply= 750;
%bbci.adaptation.policy = 'pmean';
%bbci.adaptation.running= 0;
%bbci.adaptation.UC= 0.05;
%bbci.adaptation.adaptation_ival = [0 20000]
%bbci.adaptation.load_tmp_classifier= 1;
%bbci.setup_opts.usedPat= 'auto';

global bbci_default
bbci_default= bbci;