if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/bfnt_a3_season2'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to BFNT-A3 - Season 2\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

bvr_sendcommand('loadworkspace', 'reducerbox_64std_bfnt_a3_season2');


% Funktioniert der ckeck?
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%ppTrigger(1);


global TODAY_DIR
acq_makeDataFolder;

VP_SCREEN= get(0,'ScreenSize');
%VP_SCREEN= [-1919 0 1920 1200];
fprintf('Type ''run_bfnt_a3_season2'' and press <RET>.\n');
