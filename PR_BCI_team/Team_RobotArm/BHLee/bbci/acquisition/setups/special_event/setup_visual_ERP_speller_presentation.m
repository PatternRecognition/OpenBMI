VP_CODE = 'Temp';
addpath([BCI_DIR 'acquisition/setups/special_event']);

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
% bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu'); % used for ErrP calibration study
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids'); 

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('multiple_folders', 1);
%mkdir([TODAY_DIR 'data']);
VP_SCREEN = [0 0 1920 1180];
% VP_SCREEN = [0 0 1280 1024]; % eyetracker Bildschirm


%% Online settings
general_port_fields.feedback_receiver= 'pyff';
COPYSPELLING_FINISHED = 246;
bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= 'ERP_Speller';
bbci.classDef= {[31:49], [11:29];
                'target', 'nontarget'};
bbci.start_marker= [252];
bbci.quit_marker= [4 5 COPYSPELLING_FINISHED];
bbci.pause_after_quit_marker = 0;
bbci_default = bbci;
