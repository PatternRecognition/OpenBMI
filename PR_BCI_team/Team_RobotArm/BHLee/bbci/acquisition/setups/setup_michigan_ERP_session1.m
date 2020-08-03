if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/michigan']);
% addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\n\nWelcome to P300 HexoSpeller \n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_16ch_VisualSpeller');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR

acq_makeDataFolder('multiple_folders', 1);

%VP_SCREEN = [0 10 1920 1180];
VP_SCREEN = [0 0 1280 1024]; % eyetracker Bildschirm

fprintf('Type ''run_michigan_ERP_session1'' and press <RET>\n');

%% Online settings
RUN_END = 246;
NR_SEQUENCES= 5;

bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= 'ERP_Speller';
bbci.classDef= {[31:49], [11:29];
                'target', 'nontarget'};
bbci.start_marker= [240];           %% Countdown start
bbci.quit_marker= [2 4 RUN_END];  %% Push button or end markers
bbci.pause_after_quit_marker= 1000;
startup_new_bbci_online;
