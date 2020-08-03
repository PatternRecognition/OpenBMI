%% InputReader test
% This block tests if the input reader finds the wheel and the pedal for
% the data recording
fprintf('Testing the InputReader.\n');
fprintf('Press <RETURN> to start InputReader test.\n');
pause; fprintf('Ok, starting...\n');
r = dos('C:\bbci\torcs-1.3.1\runtime\InputReader.exe test');
if(r ~= 0)
  error('The InputReader could not find the wheel or the pedal please check the usb connections or restart the system.');
else
  fprintf('Input Reader can read from wheel and pedal. Press <RETURN> to continue.\n');
  pause; fprintf('Ok, ...\n');
end

%% Prepare cap

bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

TORCS_DIR = 'C:\bbci\torcs-1.3.1\runtime';
SETTINGS_FILE = 'bbci\settings.txt';


%% -newblock
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

% %% Brake oddball test
% setup_carrace_season1_braking;
% fprintf('Press <RETURN> to TEST brake-oddball.\n');
% pause; fprintf('Ok, starting...\n');
% stim_oddballVisual(10, opt, 'test',1);
% fprintf('Press <RETURN> to start brake-oddball.\n');
% pause; fprintf('Ok, starting...\n');
% stim_oddballVisual(N, opt);

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

% %% Brake oddball with EEG
% setup_carrace_season1_braking;
% fprintf('Press <RETURN> to start brake-oddball.\n');
% pause; fprintf('Ok, starting...\n');
% stim_oddballVisual(N, opt);
% pause(2);

% %% Self paced breaking with EEG
% stimutil_fixationCross;
% 
% fprintf('Press <RETURN> to start self-paced braking.\n');
% pause; 
% fprintf('Starting Local InputReader.\n');
% dos('C:\bbci\torcs-1.3.1\runtime\InputReader.exe local BreakSync C:\bbci\torcs-1.3.1\runtime\ &');
% fprintf('Ok, starting...\n');
% fprintf('Ok.\n');
% pause(1);
% cd(TODAY_DIR)
% bvr_startrecording('selfpaced_braking');
% pause(6*60 + 10);
% bvr_sendcommand('stoprecording');
% fprintf('Press <RETURN> to stop the InputReader.\n');
% pause;fprintf('Ok, stoping...\n');
% system('taskkill /F /IM InputReader.exe');
% fprintf('Ok.\n');

% fprintf('Enter VP code "%s" in settings file.\nPress <RETURN> when done.', VP_CODE);
% system(['notepad ' TORCS_DIR '\' configFolder '\bbci\settings.txt']);
% pause

%% - OBSERVATION
desc= stimutil_readDescription('carrace_observation');
h_desc= stimutil_showDescription(desc, 'waitfor',0,'clf',1);
fprintf('Press <RET> to proceed.\n'); 
pause,close all

%% -observation newblock
configFolder = 'season2config_observation'; soundfile = '';
load_config_and_start_torcs;

fprintf('Start TORCS new race.\n');
fprintf('Subject should count brakelight occurences.\n');
fprintf('Press <RETURN> to start carrace observation run 1.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_observation');
pause(7*60);
bvr_sendcommand('stoprecording');
pause(5);
[cnt, mrk] = eegfile_readBV([TODAY_DIR 'carrace_observation'], 'clab', {'Cz', 'Pz'});
fprintf('Correct #occurences: %d\n', sum(ismember(mrk.desc, 'S  8')));

%% -observation newblock
fprintf('Press <RET> to proceed.\n'); 
pause;
configFolder = 'season2config_observation'; soundfile = '';
load_config_and_start_torcs;

fprintf('Start TORCS new race.\n');
fprintf('Subject should count brakelight occurences.\n');
fprintf('Press <RETURN> to start carrace observation run 2.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_observation');
pause(8*60);
bvr_sendcommand('stoprecording');
pause(5);
[cnt, mrk] = eegfile_readBV([TODAY_DIR 'carrace_observation02'], 'clab', {'Cz', 'Pz'});
fprintf('Correct #occurences: %d\n', sum(ismember(mrk.desc, 'S  8')));

%% - DRIVING: Instruction, load configuration and start torcs newblock 
desc= stimutil_readDescription('carrace_drive');
h_desc= stimutil_showDescription(desc, 'waitfor',0,'clf',1);
fprintf('Press <RET> to proceed.\n'); 
pause,close all

%% -Starting server InputReader
fprintf('Starting Server InputReader.\n');
dos('C:\bbci\torcs-1.3.1\runtime\InputReader.exe server &');
fprintf('Ok.\n');

%% -driving newblock (1)
fprintf('Press <RET> to proceed.\n'); 
pause;
configFolder = 'season2config'; soundfile = 'daimlerSounds_block1.txt';
load_config_and_start_torcs;

fprintf('Start TORCS new race.\n');
fprintf('Press <RETURN> to start carrace run 1.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_drive');
pause(48*60);
bvr_sendcommand('stoprecording');
euros(:, 1) = bezahlung('');
fprintf('Current bonus: %f EUR\n', sum(sum(euros)));

%% -driving newblock (2)
fprintf('Press <RET> to proceed.\n'); 
pause;
configFolder = 'season2config'; soundfile = 'daimlerSounds_block2.txt';
load_config_and_start_torcs;

fprintf('Start TORCS new race.\n');
fprintf('Press <RETURN> to start carrace run 2.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_drive');
pause(48*60);
bvr_sendcommand('stoprecording');
euros(:, 2) = bezahlung('02');
fprintf('Current bonus: %f EUR\n', sum(sum(euros)));

%% -driving newblock (3)
fprintf('Press <RET> to proceed.\n'); 
pause;
configFolder = 'season2config'; soundfile = 'daimlerSounds_block3.txt';
load_config_and_start_torcs;

fprintf('Start TORCS new race.\n');
fprintf('Press <RETURN> to start carrace run 3.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_drive');
pause(48*60);
bvr_sendcommand('stoprecording');
euros(:, 3) = bezahlung('03');
fprintf('Current bonus: %f EUR\n', sum(sum(euros)));

save([TODAY_DIR 'bezahlung'], 'euros');
