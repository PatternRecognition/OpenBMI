

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/projekt_hausler09'], path);

fprintf('\n\nWelcome to the Project Tactile P300 \n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
%bvr_sendcommand('loadworkspace', 'V-Amp_visual_P300');
%bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu');

% try
%   bvr_checkparport('type','S');
% catch
%   error('Check amplifiers (all switched on?) and trigger cables.');
% end


%VP_SCREEN= [-1023 0 1024 768];
