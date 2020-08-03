if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
  pause(4)
end

addpath([BCI_DIR 'acquisition/setups/TOBI_dry_test_2011']);

fprintf('\n\nWelcome to TOBI dry cap test 2011\n\n');

%% setup brainvision stuff
% system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
%bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', ['Nouz_ActiCAP19_parallel']); 


pause(1);  %check parallel port
% try
%   bvr_checkparport('type','S');
% catch
%   error('Check amplifiers (all switched on?) and trigger cables.');
% end

%% globals
global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];
bbci= [];

%% setup udp

% bvr_sendcommand('viewsignals');
pause(1)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

addpath('C:\Dokumente und Einstellungen\sysadm\Desktop\Nouzz\bv2nouzz')
nouzz_connect;
nouzz_sendcommand('stoprecording');

