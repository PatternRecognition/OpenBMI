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
    
%% short example trial of standard P300 test
N=20;
clear opt;
setup_spatialbci_TRAIN;
opt.perc_dev = 20/100;
opt.avoid_dev_repetitions = 1;
opt.require_response = 0;
opt.isi = 1000;
opt.fixation = 1;
opt.speech_intro = '';
opt.fixation =1;
opt.msg_fin = 'Ende'
opt.msg_intro = 'Entspannen';
opt.speech_dir = 'C:\svn\bbci\acquisition\data\sound\german\upSampled';
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;

stim_oddballAuditory(N, opt);


%% do the standard P300 test
N=250;
iterations = 2;
clear opt;
setup_spatialbci_TRAIN;
opt.perc_dev = 20/100;
opt.avoid_dev_repetitions = 1;
opt.require_response = 0;
opt.bv_host = 'localhost';
opt.isi = 1000;
opt.fixation = 1;
opt.filename = 'oddballStandardMessung';
opt.speech_intro = '';
opt.fixation =1;
opt.msg_fin = 'Ende'
opt.msg_intro = 'entspannen';
opt.speech_dir = 'C:\svn\bbci\acquisition\data\sound\german\upSampled';
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;
%testing
% opt.bv_host = '';
% opt.filename = '';
% opt.test = 1;

for i = 1:iterations,
  stim_oddballAuditory(N, opt);
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end
  
%% Short example of the offline runs
close all;
desc= stimutil_readDescription('spatial_online_train');
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to continue.');

clear opt;

setup_spatialbci_TRAIN;
opt.useSpeech=1;
VP_SCREEN = position;
opt.position=position;
volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];
opt.calibrated=diag(volume);
opt.nrRounds = 1;
opt.maxRounds = 15;
opt.requireResponse = 1;
opt.nrExtraStimuli = 3;
opt.responseOffset = 100;
opt.bv_host = '';
opt.filename = '';
opt.test = 1;
opt.doLog = 0;

auditory_MainRoutine(opt, 'trial_pause', 1);

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
setup_AEP_online; edit bbci_bet_analyze_AEP.m;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;
% this calculates the training error for all directions 
% and plots target versus non-target conditions for each direction
AEP_visualize_directions;

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

%% determine sentence to be used
senIdx = randperm(2);
senIdx = senIdx(1);

%% do the spelling run
clear opt;
setup_spatialbci_HEXO;
opt.useSpeech=1;
VP_SCREEN = position;
opt.position=position;
spellString{1} = {'Franz' 'jagt' 'im' 'Taxi' 'quer' 'durch' 'Berlin'};
spellString{2} = {'Sylvia' 'wagt' 'quick' 'den' 'Jux' 'bei' 'Pforzheim'};
spellString{3} = {'_'};
opt.visualize_hexo = 0;
opt.visualize_text = 1;
opt.dataPort=12345;
volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];
opt.calibrated=diag(volume);
% % test settings
% opt.test = 1;
% opt.debugClasses = 1;
% opt.maxRounds = 2;


close all;
settings_bbci= {'quit_marker', 254};
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s'';set_general_port_fields(''localhost'');', VP_CODE, TODAY_DIR);
bbci_cfy= [TODAY_DIR 'bbci_classifier.mat'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''')'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']); 
pause(10);    
for i = 1:length(spellString{senIdx}), % fuer jedes Wort
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


% % %% do the predictive iterations run
% % clf;
% % clear opt;
% % setup_spatialbci_HEXO;
% % opt.sayLabels=0;
% % VP_SCREEN = position;
% % opt.position=position;
% % opt.spellString = 'Franz jagt im komplett verwahrlosten Taxi';
% % opt.visualize_hexo = 1;
% % opt.visualize_text = 1;
% % opt.sayLabels = 0;
% % opt.itType = 'adaptive';
% % % test settings
% % opt.test = 1;
% % opt.debugClasses = 1;
% % opt.maxRounds = 2;
% % 
% % desc= stimutil_readDescription('spatial_online_stopIterations');
% % stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to continue.');
% % 
% % auditory_mainRoutine(opt, 'trial_pause', 5, 'sendResultTrigger', 1);
% % 
% %  

