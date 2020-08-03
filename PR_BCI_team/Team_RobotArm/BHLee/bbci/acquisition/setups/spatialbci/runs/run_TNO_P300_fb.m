%% set everything up
warning('turn the volume all the way up on the m-audio card and press enter to proceed');
input('');
if ~exist(VP_CODE) || isempty(VP_CODE),
  VP_CODE = input('Give VP code please. ', 's');
end

global VP_SCREEN;
position = [-1280 0 1280 1024];

volume = [0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0];

bvr_sendcommand('loadworkspace', ['gTec_14_p3_tactile']);

% Set condition settings
stimDuration = [0 200 4000];
fileSuffix = {'NoFB', ['FBms' num2str(stimDuration(2))], 'FBAlwaysOn'};


    
%% short example trial of standard P300 test
N=20;
clear opt;
setup_spatialbci_TRAIN;
VP_SCREEN = position;
opt.position=position;
opt.handle_background= stimutil_initFigure(opt);
opt.perc_dev = 20/100;
opt.avoid_dev_repetitions = 1;
opt.require_response = 0;
opt.isi = 1000;
opt.fixation = 1;
opt.speech_intro = '';
opt.fixation =1;
opt.msg_fin = 'Einde'
opt.msg_intro = 'Ontspannen';
opt.speech_dir = 'C:\svn\bbci\acquisition\data\sound\german\upSampled';
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;
opt.speech_intro = '';
%testing
opt.bv_host = '';
opt.filename = '';
opt.test = 1;

stim_oddballAuditory(N, opt);


%% do the standard P300 test
N=150;
iterations = 2;
clear opt;
setup_spatialbci_TRAIN;
VP_SCREEN = position;
opt.position=position;
opt.handle_background= stimutil_initFigure(opt);
opt.perc_dev = 20/100;
opt.avoid_dev_repetitions = 1;
opt.require_response = 0;
opt.bv_host = 'localhost';
opt.isi = 1000;
opt.fixation = 1;
opt.filename = 'oddballStandardMessung';
opt.speech_intro = '';
opt.fixation =1;
opt.msg_fin = 'Einde'
opt.msg_intro = 'Ontspannen';
opt.speech_dir = 'C:\svn\bbci\acquisition\data\sound\german\upSampled';
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;
opt.speech_intro = '';
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
clear opt;

setup_spatialbci_TACTILE;
opt.appPhase = 'train';
opt.itType = 'adaptive';
opt.adapt_iter_func = 'adapt_trigger_stim';
opt.adapt_iter_param = '{''clsOut'', tmpClass}';
opt.tact_duration = 200;
opt.randomClass = 1;
opt.minRounds = 4;

opt.useSpeech=0;
VP_SCREEN = position;
opt.position=position;
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
close all;
clear opt;

setup_spatialbci_TACTILE;
opt.appPhase = 'train';
opt.itType = 'adaptive';
opt.adapt_iter_func = 'adapt_trigger_stim';
opt.adapt_iter_param = '{''clsOut'', tmpClass}';
opt.randomClass = 1;
opt.minRounds = 4;

opt.useSpeech=0;
VP_SCREEN = position;
opt.position=position;

opt.calibrated=diag(volume);
opt.nrRounds = 2;
opt.maxRounds = 15;
opt.requireResponse = 1;
opt.nrExtraStimuli = 3;
opt.responseOffset = 100;

bvr_startrecording('impDummy','impedances',1);
bvr_sendcommand('stoprecording');
opt.impedanceCheck = 0;

% % %DELETE THIS
% % opt.requireResponse = 0;
% % %END DELETE THIS

% start the actual calibration trials
auditory_MainRoutine(opt, 'tact_duration', 0);
stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
conditions = randperm(length(stimDuration));
for i=1:length(conditions),
  disp(sprintf('Doing %i ms condition', stimDuration(conditions(i))));
  auditory_MainRoutine(opt, 'tact_duration', stimDuration(conditions(i)));
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end


%% run the analysis
setup_AEP_tact_online;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;


%% do the online runs
clear opt;
setup_spatialbci_TACTILE;
opt.appPhase = 'feedback';
opt.itType = 'adaptive';
opt.adapt_iter_func = 'adapt_trigger_stim';
opt.adapt_iter_param = '{''clsOut'', tmpClass}';
opt.minRounds = 4;

opt.useSpeech=0;
VP_SCREEN = position;
opt.position=position;
opt.dataPort=12345;
opt.calibrated=diag(volume);
opt.maxRounds = 15;
opt.requireResponse = 1; %%% SET BACK TO 1
opt.responseOffset = 100;

runsPerCondition = 3;

if ~exist('condSeq') || isempty(condSeq),
  condSeq = [];
  for i = 1:runsPerCondition,
    condSeq = [condSeq randperm(length(stimDuration))];
  end
end

close all;
settings_bbci= {'quit_marker', 254};
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s'';set_general_port_fields(''localhost'');', VP_CODE, TODAY_DIR);
bbci_cfy= [TODAY_DIR 'bbci_classifier.mat'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''')'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']); 
pause(10);

for i = 1:length(condSeq),
  condId = condSeq(i);
  opt.filename = ['OnlineRunFile' fileSuffix{condId}];
  bvr_startrecording([opt.filename VP_CODE], 'impedances', 0);  
  try
    get_data_udp;
  end
  opt.tact_duration = stimDuration(condId);
  auditory_MainRoutine(opt, 'trial_pause', 1, 'sendResultTrigger', 1, 'recorderControl', 0);
  bvr_sendcommand('stoprecording');
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end

%% helper calls
fclose(fopen('all')); % remove the handles to log files. Allows for deletion of the file.
