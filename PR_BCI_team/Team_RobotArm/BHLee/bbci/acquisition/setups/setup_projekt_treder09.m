if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/projekt_treder09'], path);

fprintf('\n\nWelcome to the Project Visual P300 beyond fixation.\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
%bvr_sendcommand('loadworkspace', 'V-Amp_visual_P300');
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;

%VP_SCREEN= [-1023 0 1024 768];
fprintf('External display needs to be primary display.\n');
fprintf('Display resolution of external display must be set to 1280x1024.\n');
fprintf('Type ''run_projekt_treder09'' and press <RET>.\n');