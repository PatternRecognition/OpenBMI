AUTOMATIC_EQ_TRIALS_PER_CLASS = int16(1)
AUTOMATIC_DIFF_TRIALS_PER_CLASS = int16(2)  
STIM_ALL = int16(0)
STIM_LEFT = int16(1)
STIM_RIGHT = int16(2)
STIM_FEET = int16(3)

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks']);

fb= struct();
fb.offline= int16(1);
fb.debug= int16(0);

fb.screenPos = int16([int16(1900) int16(0) int16(1920) int16(1200)]);

pyff('init','FeedbackCursorArrowFES');
pyff('set',fb);
  
%% TEST

fb.classes =  {'left', 'right', 'foot'}
fb.classesDirections =  {'left', 'right', 'foot'}
fb.classesStimulation = [int16(-1), int16(-1), int16(-1)]
fb.calibration = int16(1)
fb.trialsPerClass = int16(6)
fb.stimulation_mode = int16(4)
fb.durationActivationPeriod=int16(3000)
fb.trialGenerationPolicy = AUTOMATIC_EQ_TRIALS_PER_CLASS
fb.basename = 'MI_ONLY'

pyff('init','FeedbackCursorArrowFES');
pyff('set',fb);


pyff('play');
fb.calibration = int16(1)
fb.classes =  {'left', 'right'}
fb.classesDirections =  {'left', 'right'}
fb.classesStimulation = [int16(1), int16(2)]
fb.stimulation_mode = int16(0)
fb.trialsPerClass = int16(50)
fb.durationActivationPeriod=int16(3000)

fprintf('Press <RETURN> when ready.\n');
pause
pyff('init','FeedbackCursorArrowFES');
pyff('set',fb);
pyff('play');

    
%% MI
% 3 runs MI (25 trials per class per run = 75 trials per class) (40 min)

fb.classes =  {'left', 'right', 'foot'}
fb.classesDirections =  {'left', 'right', 'foot'}
fb.trialGenerationPolicy = AUTOMATIC_EQ_TRIALS_PER_CLASS
fb.classesStimulation = [int16(-1), int16(-1), int16(-1)]
fb.trialsPerClass = int16(25)
fb.stimulation_mode = int16(4)
fb.TODAY_DIR = TODAY_DIR;

for i=1:3,
  fprintf('Press <RETURN> when ready.\n');
  pause

    pyff('init','FeedbackCursorArrowFES');
    pyff('set',fb);

    pyff('play', 'basename', 'MI_ONLY', 'impedances',0);
    stimutil_waitForMarker({['S' num2str(254)]});
    pyff('quit');
end



%% Analyze data and extract best two classes
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto');
bbci.setup='cspauto';

%bbci.subdir = 'D:\data\bbciRaw\VP14m_11_06_22';
bbci.subdir = TODAY_DIR;
bbci.train_file= strcat(bbci.subdir, '/MI_ONLY*');

bbci.setup_opts.selband_opt.band = [6 28];
bbci.setup_opts.selband_opt.band_topscore = [7 28];

bbci.setup_opts.selival_opt.max_ival = [500 3200];
bbci.setup_opts.selival_opt.start_ival = [750 2500];
bbci.setup_opts.selival_opt.default_ival = [1000 2500];

bbci_bet_prepare
bbci_bet_analyze

bbci.adaptation.running = 1;
bbci.adaptation.load_tmp_classifier = 1;
bbci.adaptation.verbose = 2;
bbci.adaptation.mrk_end = [101];

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish



%% FES
% 2 runs of FES on two limbs ( 50 trials per class per run = 100 trials per class) ( min)
fb.classes =  {'left', 'right'}
fb.classesDirections =  {'left', 'right'}
fb.classesStimulation = [int16(1), int16(2)]
fb.stimulation_mode = int16(0)
fb.trialsPerClass = int16(50)
fb.TODAY_DIR = TODAY_DIR;

for i=1:2,
    fprintf('Press <RETURN> when ready.\n');
    pause

    pyff('init','FeedbackCursorArrowFES');
    pyff('set',fb);
    pyff('play', 'basename', 'FES_ONLY', 'impedances',0);
    stimutil_waitForMarker({['S' num2str(254)]});
    pyff('quit');
end

%% MI + FES

% 4 runs, 150 trials per run
% if right and foot:
% 1) 50 trials MI 2 classes without fes
% 2) 50 trials FES right hand (fex), while left and foot trials
% 3) 50 trials FES left hand (fex), while foot trials

fb.trialGenerationPolicy = AUTOMATIC_DIFF_TRIALS_PER_CLASS
fb.classes =  {'right', 'foot', 'right FES left', 'foot FES left', 'right FES right', 'foot FES right'}
fb.classesDirections =  {'right', 'foot', 'right', 'foot', 'right', 'foot'}
fb.classesStimulation = [-1, -1, STIM_LEFT, STIM_LEFT, STIM_RIGHT, STIM_RIGHT]
fb.numClassTrials = [int16(25),int16(25),int16(25),int16(25), int16(25), int16(25)]
fb.pauseAfter = int16(30)
fb.TODAY_DIR = TODAY_DIR;

% fb.trialGenerationPolicy = AUTOMATIC_DIFF_TRIALS_PER_CLASS
% fb.classes =  {'left', 'foot', 'left FES right', 'foot FES right', 'left FES left', 'foot FES left'}
% fb.classesDirections =  {'left', 'foot', 'left', 'foot', 'left', 'foot'}
% fb.classesStimulation = [-1, -1, STIM_RIGHT, STIM_RIGHT, STIM_LEFT, STIM_LEFT]
% fb.numClassTrials = [int16(25),int16(25),int16(25),int16(25), int16(25), int16(25)]
% fb.pauseAfter = int16(30)

for i=1:3,    
    pyff('init','FeedbackCursorArrowFES');
    pyff('set',fb);
    pyff('play', 'basename', 'MI_AND_FES', 'impedances',0);
    stimutil_waitForMarker({['S' num2str(254)]});
    pyff('quit');
end

pyff('quit');
    