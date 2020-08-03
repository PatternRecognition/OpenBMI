if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/motionVEP']);
% addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\n\nWelcome to motion VEP.\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('multiple_folders', 1);
%mkdir([TODAY_DIR 'data']);
%VP_SCREEN = [0 10 1920 1180];
VP_SCREEN = [0 0 1280 1024]; % eyetracker Bildschirm

fprintf('Type ''run_motionVEP'' and press <RET>\n');

%% Online settings
general_port_fields.feedback_receiver= 'pyff';
COPYSPELLING_FINISHED = 246;

bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= 'ERP_Speller';
bbci.classDef= {[31:49], [11:29];
                'target', 'nontarget'};
bbci.start_marker= [240];
bbci.quit_marker= [2 4 8 COPYSPELLING_FINISHED];
bbci.pause_after_quit_marker = 0;
bbci_default = bbci;
