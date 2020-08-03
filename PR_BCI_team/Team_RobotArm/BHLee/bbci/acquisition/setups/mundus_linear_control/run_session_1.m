
%% PYFF
%pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0 );

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'dir', 'D:\development\2011.5\src', 'bvplugin',0 );

fb= struct();
fb.debug= int16(0);

 
    
%% CALIBRATION
% 3 runs MI (25 trials per class per run = 75 trials per class) (40 min)
fb.calibration = int16(1);
fb.classes =  {'left', 'right', 'foot'}
fb.classesDirections =  {'left', 'right', 'foot'}
fb.trialGenerationPolicy = int16(1)
fb.classesStimulation = [int16(-1), int16(-1), int16(-1)]
fb.trialsPerClass = int16(25)

fb.classTrialsPerPause = [int16(21), int16(21), int16(21), int16(12)]
fb.pauseAfter = int16(21);
fb.classesStimulation = [int16(-1), int16(-1), int16(-1)] 
    
fb.stimulation_mode = int16(4)
fb.durationActivationPeriod = int16(3000)

filename = 'MI_cl_'

for i=1:3,

  pyff('init','FeedbackCursorArrowFES');
  pyff('set',fb);
  
  pause(1);
  pyff('play', 'basename',filename, 'impedances',0);
  %pyff('play');
  stimutil_waitForMarker({['S' num2str(254)]});
  pyff('stop');
end



%% Analyze data and extract best two classes

bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
%bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto');
bbci.setup='cspauto';
%bbci.subdir = 'D:\data\bbciRaw\VP17m_11_07_25';
bbci.train_file= strcat(bbci.subdir, '/MI_cl*');
        

bbci_bet_prepare

bbci.setup_opts.selband_opt.band = [6 28];
bbci.setup_opts.selband_opt.band_topscore = [7 28];
bbci.setup_opts.selival_opt.max_ival = [500 3200];
bbci.setup_opts.selival_opt.start_ival = [750 2500];
bbci.setup_opts.default_ival = [1000 2500];    
bbci.setup_opts.visu_ival = [500 4000];
        
bbci_bet_analyze



% extra data to save
bbci.adaptation.running = 1;
bbci.adaptation.load_tmp_classifier = 1;
bbci.adaptation.verbose = 2;
bbci.adaptation.mrk_end = [11,12,21,22,31,32];

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

bbci.adaptation.running = 1;
bbci.adaptation.load_tmp_classifier = 1;

% save the data into a file in the same dir
bbci_bet_finish


pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0);

%% FEEDBACK
% 2 runs of two limbs ( 50 trials per class per run = 100 trials per class) ( min)

%% VR Selection parameters
fb.threshold1 = single(-0.24);
fb.threshold2 = single(0.27);
fb.selectTime1 = int16(800);
fb.selectTime2 = int16(800);

%% MI feedback parameters
fb.classes =  {'left', 'right'}
fb.classesDirections =  {'left', 'right'}
fb.trialGenerationPolicy = int16(1)
fb.classesStimulation = [int16(-1), int16(-1)]
fb.stimulation_mode = int16(0)
fb.trialsPerClass = int16(25)

fb.classTrialsPerPause = [int16(18), int16(18), int16(14)]
fb.pauseAfter = int16(18);


fb.calibration = int16(0)
fb.pause = int16(0)
fb.feedbackWithVR = int16(1) 



fb.introDuration = int16(10000)
fb.shortPauseCountdownFrom = int16(15)

fb.pauseDuration = int16(15000)
fb.countdownFrom = int16(10)
fb.shortPauseCountdownFrom = int16(15)

fb.feedbackWithVR= int16(0) 
log_filename = [TODAY_DIR 'practice_' VP_CODE '.log'];
 
for i=1:2,
    fb.screenPos = int16([int16(-200) int16(-200) int16(1600) int16(1000)]);

    pyff('init','FeedbackCursorArrowVR');
    pyff('set',fb);
    
    pause(1); 
    pyff('play', 'basename', 'MI_fb_', 'impedances',0);
    bbci_bet_apply([TODAY_DIR 'bbci_classifier_cspauto'], 'bbci.feedback','1d', 'bbci.fb_port', 12345);
    pause(1);
    pyff('quit');
end
    