%% student group (November 2011) that investigates spatial auditory
%% attention and interval selection

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
  fprintf('\n\nPress Ctrl-C to stop')
  pause(5)
end

acquire_func = @acquire_bv;
setup_bbci_online; %% needed for acquire_bv

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';

fprintf('\n\nWelcome to the setup of the auditory steady state experiment!\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');


bvr_sendcommand('loadworkspace', 'reducerbox_64std');

pause(1)


try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('log_dir',1);

tmpLogDir = [TODAY_DIR 'log\'];
tmpLogDir(strfind(tmpLogDir,'/')) = '\'; %Python doesn't accept '/' in a path!!


addpath('E:\svn\bbci\acquisition\setups\auditory_steady_state_SSCA_data')


