% Auditory ISI-experiment
% @JohannesHoehne 

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
  fprintf('\n\nPres Ctrl-C to stop')
end

setup_bbci_online; %% needed for acquire_bv

set_general_port_fields('localhost');

fprintf('\n\nWelcome to ISI experiment of the BCI students project\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');


bvr_sendcommand('loadworkspace', 'reducerbox_64std_MastoidAuditory.rwksp'); 

pause(2)


fprintf('While steeing up the cap, we should explain the setup to the subject !!\n');


try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('log_dir',1);

addpath([BCI_DIR '\acquisition\setups\TOBI_students_isi'])
fprintf('everything was successfully set up! \n')
fprintf('Type ''run_TOBI_students_isi'' and press <RET>.\n');
