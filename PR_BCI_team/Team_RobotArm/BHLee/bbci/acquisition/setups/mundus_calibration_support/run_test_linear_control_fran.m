
AUTOMATIC_EQ_TRIALS_PER_CLASS = int16(1)
AUTOMATIC_DIFF_TRIALS_PER_CLASS = int16(2)  
STIM_ALL = int16(0)
STIM_LEFT = int16(1)
STIM_RIGHT = int16(2)
STIM_FEET = int16(3)



%send_xmlcmd_udp('init', '127.0.0.1', 12345);
%pyff('startup', 'dir','D:\svn\pyff\src', 'a', 'D:\svn\bbci\python\pyff\src\Feedbacks', 'gui', 1);
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0 );
fb= struct();
%fb.offline= int16(0);
fb.calibration = int16(0);
fb.debug= int16(0);
fb.screenPos = int16([int16(0) int16(0) int16(2560) int16(200)]);


pyff('init','FeedbackCursorArrowVR');
pyff('set',fb);
 
    
%% MI
% 3 runs MI (25 trials per class per run = 75 trials per class) (40 min)

fb.classes =  {'left', 'right', 'foot'}
fb.classesDirections =  {'left', 'right', 'foot'}
fb.trialGenerationPolicy = AUTOMATIC_EQ_TRIALS_PER_CLASS
fb.classesStimulation = [int16(-1), int16(-1), int16(-1)]
fb.trialsPerClass = int16(25)
fb.stimulation_mode = int16(4)

filename = 'MI_cl_'

for i=1:3,

  pyff('init','FeedbackCursorArrowVR');
  pyff('set',fb);
  
  pause(1);
  %pyff('play', 'basename',filename, 'impedances',0);
  pyff('play');
  stimutil_waitForMarker({['S' num2str(254)]});
  pyff('stop');
end



%% Analyze data and extract best two classes
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto');
bbci.setup='cspauto';
bbci.train_file= strcat(bbci.subdir, '/MI_ONLY*');

bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

bbci.adaptation.running = 1;
bbci.adaptation.load_tmp_classifier = 1;

bbci_bet_finish


pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0);
%% FBACK
% 2 runs of two limbs ( 50 trials per class per run = 100 trials per class) ( min)

%% VR Selection parameters
fb.threshold1 = single(-0.8);
fb.threshold2 = single(0.5);
fb.selectTime1 = int16(2000);
fb.selectTime2 = int16(2500);

%% MI feedback parameters
fb.classes =  {'left', 'right'}
fb.classesDirections =  {'left', 'right'}
fb.trialGenerationPolicy = AUTOMATIC_EQ_TRIALS_PER_CLASS
fb.classesStimulation = [int16(-1), int16(-1)]
fb.stimulation_mode = int16(0)
fb.trialsPerClass = int16(25)
fb.calibration = int16(0)
fb.pause = int16(0)
fb.feedbackWithVR = int16(1) 
fb.TODAY_DIR = TODAY_DIR;
fb.introDuration = int16(10000)
fb.pauseDuration = int16(10000)
fb.countdownFrom = int16(10)

log_filename = [TODAY_DIR 'practice_' VP_CODE '.log'];

 
for i=1:2,
    pyff('init','FeedbackCursorArrowVR');
    pyff('set',fb);
    pause(1); 
    pyff('play', 'basename', 'MI_fb_', 'impedances',0);
    bbci_bet_apply([TODAY_DIR 'bbci_classifier_cspauto'], 'bbci.feedback','1d', 'bbci.fb_port', 12345);
    pause(1);
    pyff('quit');
se
end
    