if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/projekt_biomed08'], path);

fprintf('\n\nWelcome to Projekt Biomed 08\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
%bvr_sendcommand('loadworkspace', 'projekt_biomed08');
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu');
try,
  bvr_checkparport('type','S');
catch
  error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
end  

global TODAY_DIR
acq_makeDataFolder;

VP_SCREEN= [-1023  243  1024  749];
%-1279 0 1280 1024];
fprintf('Display resolution of secondary display must be set to 1280x1024.\n');
