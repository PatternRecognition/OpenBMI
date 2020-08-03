%% eyes open/close

sprintf('\n \n \n \n EYES CLOSED, press <ENTER> to start')
pause;
bvr_startrecording(['eyesClosed_' VP_CODE], 'impedances', 0);
nouzz_startrecording(['nouzz_eyesClosed_' VP_CODE]);
pause(2)
ppTrigger(253)
pause(30);
ppTrigger(254)
bvr_sendcommand('stoprecording');
nouzz_sendcommand('stoprecording');


disp('EYES OPEN - check for fixation cross, press <ENTER> to start')
pause;

bvr_startrecording(['eyesOpen_' VP_CODE], 'impedances', 0);
nouzz_startrecording(['nouzz_eyesOpen_' VP_CODE]);
pause(2)
ppTrigger(253)
pause(30);
ppTrigger(254)
bvr_sendcommand('stoprecording');
nouzz_sendcommand('stoprecording');


%% artifact measurement

fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
nouzz_startrecording(['nouzz_' opt.filename VP_CODE]);
opt.impedances=0;
opt.checkparport=0;
stim_artifactMeasurement(seq, wav, opt);
nouzz_sendcommand('stoprecording');


%% Standard auditory oddball with ISI 1000
disp('press <ENTER> to proceed to stdAuditory')
pause;

N=180; % total number of tones per iteration
iterations = 4;
clear opt;

opt.toneDuration = 50;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

%setup_spatialbci_GLOBAL

opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'fixed';
opt.mode = 'copy';
opt.application = 'TRAIN';

opt.countdown = 0;
opt.repeatTarget = 3;

opt.perc_dev = 20/100;
opt.avoid_dev_repetitions = 1;
opt.require_response = 0;
opt.isi = 1000;
opt.filename = 'oddballStandardMessung_isi1000';
%opt.speech_intro = '';
opt.fixation =1;
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', opt.toneDuration, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', opt.toneDuration, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;
opt.checkparport = 0;
opt.impedances = 0;

for i = 1:iterations,
    nouzz_startrecording('nouzz_oddballStandardMessung_isi1000');
    stim_oddballAuditory(N, opt);
    stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
    nouzz_sendcommand('stoprecording');
end
fprintf('\n \n \nplease close the Window with the fixation cross! \n \n')
fprintf('------------------- \n \n \n')

%% start PYFF

% Init: Start Pyff
PYFF_DIR = 'D:\svn\pyff\src';
general_port_fields= struct('bvmachine','127.0.0.1',...
    'control',{{'127.0.0.1',12471,12487}},...
    'graphic',{{'',12487}});

general_port_fields.feedback_receiver= 'pyff';

PyffStarted = pyff('startup', 'gui',1, 'dir',PYFF_DIR, 'a', 'D:\svn\bbci\python\pyff\src\Feedbacks'); % Start PYFF

pause(5);
send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

%% T9 Calibration

numCalibRuns = 2 %4

% send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
for i=1:numCalibRuns
    sbj_counts = 0;

    bvr_startrecording(['T9SpellerCalibration' VP_CODE], 'impedances', 0);
    nouzz_startrecording(['nouzz_T9SpellerCalibration' VP_CODE]);

    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'T9Speller','command','sendinit');
    pause(5);
    send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [LOG_DIR 'calibration' num2str(i) '.log'] );
    send_xmlcmd_udp('interaction-signal', 'i:spellerMode', false );
    send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false );
    send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

    fprintf('\n press ENTER to start T9 calibration\n')
    pause

    pause(1)
    send_xmlcmd_udp('interaction-signal', 'command', 'play');


    while isnumeric(sbj_counts)
        %catch counts of the trials!!
        sbj_counts = input('enter the counted number (when block completed)', 's');
        if strmatch(sbj_counts,'end')
            break;      
        else
          sbj_counts = str2num(sbj_counts);
          if ~isempty(sbj_counts)
            send_xmlcmd_udp('interaction-signal', 'i:numCounts' , sbj_counts);
          end
        end
    end
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
    bvr_sendcommand('stoprecording');
    nouzz_sendcommand('stoprecording');

    fprintf('Press <RET> to continue with the next calibration run.\n');
    pause
end

%% Brisk motor execution, slow potential with GO-signal for beta ERS


pyff('init','FeedbackCursorArrow2'); % Init Feedback
pause(3)
send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

pause(2);
pyff('play'); % Start the Feedback (it will be in pause mode)

% PYFF settings
fb_opt_int.trials = 30;
fb_opt_int.durationPerTrial = 2000; % execution time
fb_opt_int.durationIndicateGoal = 3000; % preparation/cueing time 
fb_opt_int.hitMissDuration = 4000; % Pause zwischen 2 Trials
fb_opt_int.countdownFrom = 5;


%pyff('set', fb_opt);   % Send Settings, uncomment if you want to set non
% integer values
pyff('setint', fb_opt_int); % Send Settings (Interger Values)
fb_opt= []; fb_opt_int= [];

fprintf('Position feedback window on screen\n Remember VP_CODE and setupfile\\n');


% General Settings
all_classes= {'left', 'right', 'foot'};
savename = 'exec_arrow_go';     % Filenamen für ausgeführte Bewegungen


fprintf('Parameters were set, press ENTER to start experiment\\n');
pause


% ERD Run

runs = 2;
fb_opt_int.trials = 30; % Number of trails
pyff('setint', fb_opt_int); % Send Settings (Integer Values)
send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

fb_opt.g_rel = 0.0; % Staerke der Kreuzbewegung
%fb_opt.g_rel = 0.0; % Staerke der Kreuzbewegung
%fb_opt.bias = 1.0

CLSTAGlist = {'LR' 'LF' 'RF'};
ixCLS = [];
for ii = 1:runs
    ixCLS = [ixCLS  randperm(3)];
end


for iblock = 1:length(ixCLS)
    CLSTAG = CLSTAGlist{ixCLS(iblock)};

    ci1= find(CLSTAG(1)=='LRF'); ci2= find(CLSTAG(2)=='LRF');
    classes= all_classes([ci1 ci2]);

    fprintf('Starting Run with classes %s and %s.\n', char(classes(1)), char(classes(2)));

    fb_opt.pause= false;
    fb_opt.countdown= true;
    fb_opt.availableDirections= {char(classes(1)),char(classes(2))};

    bvr_startrecording([savename, CLSTAG],  'impedances',0);
    nouzz_startrecording(['nouzz_' savename CLSTAG]);
    pause(5)
%    send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );
    pyff('set', fb_opt);
    send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

    dum = input('enter "ende" when the run is completed', 's');
    while ~strcmp(dum, 'ende')
        dum = input('enter "ende" when the run is completed', 's');
    end
    nouzz_sendcommand('stoprecording');
    bvr_sendcommand('stoprecording');

    fprintf('Run Completed. \nEnter RETURN to continue\n');
    pause

end

%% PhotoBrowser experiment


send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

condition.flash_enable=false;
condition.scaling_enable=true;
condition.rotation_enable=false;
condition.invert_enable=false;
condition.mask_enable=true;


fbsettings.screen_width = 1920;                                    % window width in pixels
fbsettings.screen_height = 1200;                                    % window height in pixels
fbsettings.screen_x = -1920;                                          % x position of left top corner of the window in pixels
fbsettings.screen_y = 0;                                          % y position of left top corner of the window in pixels

%	Variables: limits/counts
fbsettings.num_blocks                              = 4;
fbsettings.num_trials                              = 12;
fbsettings.num_subtrials_per_iteration             = 4;

% 	Variables: durations and pauses (all times in MILLISECONDS)
fbsettings.startup_sleep_duration 				    = 1000; 		% pause on startup to allow the classifier time to initialise. Set to 0 to disable.
fbsettings.cue_presentation_duration 				= 2000;
fbsettings.pre_cue_pause_duration 				    = 1000;
fbsettings.post_cue_presentation_pause_duration 	= 1000;
fbsettings.inter_trial_duration 					= 4000;
fbsettings.stimulus_duration 						= 100;
fbsettings.inter_stimulus_duration 				    = 100;
fbsettings.inter_block_pause_duration 			    = 10000;
fbsettings.inter_block_countdown_duration 		    = 3000;
fbsettings.result_presentation_duration 			= 5000;

% Variables: miscellaneous
fbsettings.max_inter_score_duration 				= 1000;        % maximum time in milliseconds to allow between successive scores...
% being received
fbsettings.udp_markers_enable = true;                              % if True, activates ue,false,truetrue,true,true,false,trueP...
% markers when send_parallel is called
fbsettings.online_mode_enable = false;                             % if True, activates online mode
fbsettings.row_col_eval_enable = true;                             % if True, activates row/column mode in the sequence generator
fbsettings.display_selection_enable = false; %true                 % if True, activates the displaying of the selected object
% after each trial
fbsettings.show_empty_image_frames_enable = false;                 % if True, empty slots in the grid will still contain the standard image frame graphic
fbsettings.mask_sequence_enable = false;

% Erste Initialisierung von Pyff mit fbsettings
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300_photobrowser','command','sendinit');
pause(3);

%send fbsettings to feedback!!
fbOpts = fieldnames(fbsettings);
for optId = 1:length(fbOpts),
    if isnumeric(getfield(fbsettings, fbOpts{optId})),
        send_xmlcmd_udp('interaction-signal', ['i:' fbOpts{optId}], getfield(fbsettings, fbOpts{optId}));
    else
        send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(fbsettings, fbOpts{optId}));
    end
    pause(.005);
end

% SEND Condition_struct
cdOpts = fieldnames(condition);

for optId = 1:length(cdOpts),
    if isnumeric(getfield(condition, cdOpts{optId})),
        send_xmlcmd_udp('interaction-signal', ['i:' cdOpts{optId}], getfield(condition, cdOpts{optId}));
    else
        send_xmlcmd_udp('interaction-signal', cdOpts{optId}, getfield(condition, cdOpts{optId}));
    end
    pause(.005);
end

fprintf('Parameters were set, press ENTER to start experiment\\n');
pause

bvr_startrecording('PhotoBrowser',  'impedances',0);
nouzz_startrecording('nouzz_PhotoBrowser');
pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
dum = input('enter "ende" when the run is completed', 's');
while ~strcmp(dum, 'ende')
    dum = input('\nenter "ende" when the run is completed', 's');
end
nouzz_sendcommand('stoprecording');
bvr_sendcommand('stoprecording');



%% Sustained motor execution for alpha ERD, without GO cue
pause(5)
send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );
pause(2)
pyff('init','FeedbackCursorArrow2'); % Init Feedback
pause(1)
send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

pause(2);
pyff('play'); % Start the Feedback (it will be in pause mode)

% PYFF settings
fb_opt_int.trials = 30;
fb_opt_int.durationPerTrial = 4000; % execution time
fb_opt_int.durationIndicateGoal = 0; % preparation/cueing time 
fb_opt_int.hitMissDuration = 4000; % Pause zwischen 2 Trials
fb_opt_int.countdownFrom = 4;


%pyff('set', fb_opt);   % Send Settings, uncomment if you want to set non
% integer values
pyff('setint', fb_opt_int); % Send Settings (Interger Values)
fb_opt= []; fb_opt_int= [];

fprintf('Position feedback window on screen\n Remember VP_CODE and setupfile\\n');


% General Settings
all_classes= {'left', 'right', 'foot'};
savename = 'exec_arrow';     % Filenamen für ausgeführte Bewegungen

fprintf('Parameters were set, press ENTER to start experiment\\n');
pause


% ERD Run

runs = 2;
fb_opt_int.trials = 30; % Number of trails
pyff('setint', fb_opt_int); % Send Settings (Integer Values)

fb_opt.g_rel = 0.0; % Staerke der Kreuzbewegung
%fb_opt.g_rel = 0.0; % Staerke der Kreuzbewegung
%fb_opt.bias = 1.0

CLSTAGlist = {'LR' 'LF' 'RF'};
ixCLS = [];
for ii = 1:runs
    ixCLS = [ixCLS  randperm(3)];
end


for iblock = 1:length(ixCLS)
    CLSTAG = CLSTAGlist{ixCLS(iblock)};

    ci1= find(CLSTAG(1)=='LRF'); ci2= find(CLSTAG(2)=='LRF');
    classes= all_classes([ci1 ci2]);

    fprintf('Starting Run with classes %s and %s.\n', char(classes(1)), char(classes(2)));

    fb_opt.pause= false;
    fb_opt.countdown= true;
    fb_opt.availableDirections= {char(classes(1)),char(classes(2))};

    bvr_startrecording([savename, CLSTAG],  'impedances',0);
    nouzz_startrecording(['nouzz_' savename CLSTAG]);
    pause(5)
%    send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );
    pyff('set', fb_opt);
    send_xmlcmd_udp('interaction-signal', 'i:udp_markers_enable', true );

    dum = input('enter "ende" when the run is completed', 's');
    while ~strcmp(dum, 'ende')
        dum = input('enter "ende" when the run is completed', 's');
    end
    nouzz_sendcommand('stoprecording');
    bvr_sendcommand('stoprecording');

    fprintf('Run Completed. \nEnter RETURN to continue\n');
    pause

end


