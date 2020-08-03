%% set everything up
warning('turn the volume all the way up on the m-audio card and press enter to proceed');
input('');
warning('Is the MARY TTS server running??');
input('');
warning('Is the proper VP_CODE set?');
input('');

global VP_SCREEN;
position = [-1920 0 1920 1200];
% small screen for testing
%warning('using VP_SCREEN for testing');

% start the recorder and load a workspace
% system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
% % % bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', ['reducerbox_64stdMastoid']);
    
 

%% do the offline runs
clear opt;
setup_spatialbci_TRAIN;
opt.useSpeech=1;
VP_SCREEN = position;
opt.position=position;
volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];
opt.calibrated=diag(volume);

blocks = 4;
opt.nrRounds = 2;
opt.maxRounds = 15;
bvr_startrecording('impDummy','impedances',1);
bvr_sendcommand('stoprecording');
opt.impedanceCheck = 0;
opt.requireResponse = 1;
opt.nrExtraStimuli = 3;
opt.responseOffset = 100;
for i=1:blocks,
  auditory_MainRoutine(opt, 'trial_pause', 3);
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end


%% run the analysis
setup_AEP_online_block2;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;


%% familiarize the subject with the paradigm
close all;
desc= stimutil_readDescription('spatial_online_fixed');
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to continue.');

clear opt;
setup_spatialbci_HEXO;
opt.useSpeech=1;
VP_SCREEN = position;
opt.position=position;
opt.spellString = 'AZ';
opt.visualize_hexo = 1;
opt.visualize_text = 1;
volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];
opt.calibrated=diag(volume);
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
opt.doLog = 0;
% opt.maxRounds = 2;
auditory_mainRoutine(opt, 'trial_pause', 1, 'debugClasses', 1);

%% Set spellstring to not be the one from the first go
senIdx = str2num(input('Give new sentence ID: ', 's'));
spellString{1} = {'Franz' 'jagt' 'im' 'Taxi' 'quer' 'durch' 'Berlin'};
spellString{2} = {'Sylvia' 'wagt' 'quick' 'den' 'Jux' 'bei' 'Pforzheim'};

%% do the first spelling run
clear opt;
setup_spatialbci_HEXO;
opt.minRounds = 4;
opt.itType = 'adaptive';
load([TODAY_DIR 'bbci_classifier.mat']);
opt.probThres = bbci.analyze.thresholds;
opt.useSpeech=1;
opt.sayLabels = 0;
opt.sayResult = 1;
opt.errorPRec = 1;
opt.errorPTrig = 200;
VP_SCREEN = position;
opt.position=position;
opt.visualize_hexo = 1;
opt.visualize_text = 1;
opt.dataPort=12345;
volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];
opt.calibrated=diag(volume);
% % test settings
% opt.test = 1;
% opt.filename = '';
% opt.randomClass = 1;
% opt.debugClasses = 1;
% opt.maxRounds = 2;

close all;
settings_bbci= {'quit_marker', 254};
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s'';set_general_port_fields(''localhost'');', VP_CODE, TODAY_DIR);
bbci_cfy= [TODAY_DIR 'bbci_classifier.mat'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''')'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']); 
pause(10);    
for i = 1:length(spellString{senIdx}),
  bvr_startrecording([opt.filename VP_CODE], 'impedances', 0);
  try
    get_data_udp;
  end
  opt.spellString = spellString{senIdx}{i};
  auditory_MainRoutine(opt, 'trial_pause', 1, 'sendResultTrigger', 1, 'recorderControl', 0);
  bvr_sendcommand('stoprecording');
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end


%% give sentence for second round
senIdx = str2num(input('Give new sentence ID: ', 's'));
spellString{1} = {'Vogel', 'Quax', 'zwickt', 'Johnys', 'Pferd', 'Bim'};
spellString{2} = {'Prall', 'vom', 'Whiskey', 'flog', 'Quax', 'den', 'Jet', 'zu', 'Bruch'};
spellString{3} = {'Das', 'Glas', 'steht', 'auf', 'dem', 'Tisch'};

%% do the second spelling run
clear opt;
setup_spatialbci_HEXO;
opt.minRounds = 4;
opt.itType = 'adaptive';
opt.filename = 'OnlineBlindRunFile';
load([TODAY_DIR 'bbci_classifier.mat']);
opt.probThres = bbci.analyze.thresholds;
opt.useSpeech=1;
opt.sayLabels = 0;
opt.sayResult = 1;
opt.errorPRec = 1;
opt.errorPTrig = 200;
VP_SCREEN = position;
opt.position=position;
opt.visualize_hexo = 0;
opt.visualize_text = 1;
opt.dataPort=12345;
volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];
opt.calibrated=diag(volume);
% spellString = input('Enter the sentence to be written: ', 's');
% spellString = {'Ich' 'denke' 'also' 'bin' 'ich'};
% DO SOMETHING WITH THE STRING
% % test settings
% opt.test = 1;
% opt.filename = '';
% opt.randomClass = 1;
% opt.debugClasses = 1;
% opt.maxRounds = 2;

close all;
% settings_bbci= {'quit_marker', 254};
% cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s'';set_general_port_fields(''localhost'');', VP_CODE, TODAY_DIR);
% bbci_cfy= [TODAY_DIR 'bbci_classifier.mat'];
% cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''')'];
% system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']); 
pause(10);    
for i = 1:length(spellString{senIdx}),
  bvr_startrecording([opt.filename VP_CODE], 'impedances', 0);
  try
    get_data_udp;
  end
  opt.spellString = spellString{senIdx}{i};
  auditory_MainRoutine(opt, 'trial_pause', 1, 'sendResultTrigger', 1, 'recorderControl', 0);
  bvr_sendcommand('stoprecording');
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end

%% helper calls
fclose(fopen('all')); % remove the handles to log files. Allows for deletion of the file.


