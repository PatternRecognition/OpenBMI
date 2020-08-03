% Online Auditory P300 Speller based on a T9 language system
% @JohannesHoehne 

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
  fprintf('\n\nPres Ctrl-C to stop')
end

setup_bbci_online; %% needed for acquire_bv

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';

fprintf('\n\nWelcome to the project for my (Johannes) Master thesis\n\n');
fprintf('ONLINE AUDITORY P300 SPELLER\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');


bvr_sendcommand('loadworkspace', 'reducerbox_64std');

pause(2)


fprintf('While steeing up the cap, it is time for the subject to get to know the stimuli !!\n');
fprintf('Start with <ENT> and cancel this run with Ctrl-C in the Console !!\n');
pause
system('cmd /C "E: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\T9Speller & python TrialPresentation.py &');

pause
fprintf('\n \n The subject should get used to the fast sequence of sounds. \n As a little example we we make some test-trials.\n');
fprintf('Start with <ENT> and cancel this run with Ctrl-C in the Console \n when subject has got used to the speed !!\n');
fprintf('Remember: EEG data is NOT recorded for this run!!! \n')
pause
system('cmd /C "E: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\T9Speller & python T9Speller.py &');
fprintf('when finished, press <ENT> \n \n')


try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('log_dir',1);

tmpLogDir = [TODAY_DIR 'log\'];
tmpLogDir(strfind(tmpLogDir,'/')) = '\'; %Python doesn't accept '/' in a path!!

%START PYFF
%system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --port=0x2030 FeedbackControllerPlugins --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &');
%system('cmd /C "E: & cd \svn\pyff\src & python FeedbackController.py -l debug -a E:\svn\bbci\python\pyff\src\Feedbacks --port=6C00')

addpath('E:\svn\bbci\acquisition\setups\T9Speller')
fprintf('PYFF was successfully set up! \n')
fprintf('Type ''run_T9calibration'' and press <RET>.\n');
