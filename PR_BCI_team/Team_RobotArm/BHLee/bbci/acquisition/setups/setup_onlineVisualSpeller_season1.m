if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/onlineVisualSpeller']);
addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\n\nWelcome to visual speller online\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_VisualSetup_linked_mastoids');

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

fprintf('Type ''run_online_visualspeller'' and press <RET>\n');

%% Online settings
COPYSPELLING_FINISHED = 246;

bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= 'ERP_Speller';
bbci.classDef= {[31:49], [11:29];
                'target', 'nontarget'};
bbci.start_marker= [];
bbci.quit_marker= [4 5 COPYSPELLING_FINISHED];
bbci_default = bbci;
