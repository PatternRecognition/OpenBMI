%% Arrow Feedback Config

fb_opt_int.g_rel =  1;
fb_opt_int.g_ab   = 1;

fb_opt_int.durationUntilBorder = 1000;
fb_opt_int.durationPerTrial = 4000;
fb_opt_int.durationIndicateGoal = 1000;
fb_opt_int.trials = 50;
fb_opt_int.pauseAfter = 50;
fb_opt_int.pauseDuration = 9000;
fb_opt_int.FPS =  60;
fb_opt_int.screenWidth =  1000;
fb_opt_int.screenHeight =  700;
fb_opt_int.countdownFrom = 2;
fb_opt_int.hitMissDuration =  1000;

fb_opt.control = 'relative';
fb_opt.fullscreen =  'True';

fb_opt.dampedMovement = 'False';
fb_opt.showPunchline = 'True';
fb_opt.damping = 'linear';
%fb_opt.damping = 'distance';
            
%fb_opt.arrowPointlist = [(0.5,0), (0.5,.33), (1,0.33), (1,0.66), (0.5,0.66), (0.5,1), (0,0.5)];
fb_opt.arrowColor = [127, 127, 127];
fb_opt.borderColor = fb_opt.arrowColor;
fb_opt.backgroundColor = [64, 64, 64];
fb_opt.cursorColor = [100, 149, 237];
fb_opt.fontColor = fb_opt.cursorColor;
fb_opt.countdownColor = [237, 100, 148];
fb_opt.punchLineColor = fb_opt.cursorColor;
fb_opt.punchLineColorImpr = [100, 200 , 100];  % if punchline is improved
        
fb_opt.punchlineThickness = 5;   % in pixels 
fb_opt.borderWidthRatio = 0.4;     % in pixels
fb_opt.punchlinePos1 = 0;
fb_opt.punchlinePos2 = 0;

fb_opt.availableDirections = {'right', 'foot'};
% fb_opt.availableDirections= {'left','right'};



%% - Preparation
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
fprintf('\nLet participant fill out first questionnaire.\n');
pause

bvr_startrecording(['impedances_beginning' VP_CODE]);
pause(1);
bvr_sendcommand('stoprecording');

% ToDo:
% Ruhemessung (Augen Auf / Zu)


% %% - Artifact measurement: Test recording
% fprintf('\n\nArtifact test run.\n');
% [seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
% fprintf('Press <RETURN> when ready to start ARTIFACT TEST measurement.\n');
% pause
% stim_artifactMeasurement(seq, wav, opt, 'test',1);
% 
% 
% %% - Artifact measurement: recording
% fprintf('\n\nArtifact recording.\n');
% %[seq, wav, opt]= setup_season10_artifacts('clstag', 'LRF');
% [seq, wav, opt]= setup_season10_artifacts('clstag', '');
% fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
% pause
% stim_artifactMeasurement(seq, wav, opt);
% fprintf('Press <RETURN> when ready to go to the RELAX measurement.\n');
% pause

%% TODO: 


%-newblock
%% - Runs 1, 2
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15;16-35] at Lap C3,4), pcovmean adaptation
%desc= stimutil_readDescription('season11_imag_fbarrow_LRF_1');
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);

%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to view the remaining instructions: ');
%desc= stimutil_readDescription('season11_imag_fbarrow_LRF_2');

%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');


nRuns= 1;
n_trials_per_run = 2;
break_every = n_trials_per_run;

setup_file= 'season11\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pcovmean_Lap_' today_str];

% here are the possible combinations of classes that are tested
tag_list= {'LR', 'LF', 'FR'};
% tag_list = {'LR'};
all_classes= {'left', 'right', 'foot'};

for ri= 1:nRuns,
    for ti= 1:length(tag_list),
        % get the class combination to test in this run
        CLSTAG= tag_list{ti};

        % this stuff we don't need if we don't use the subject independent classifier
        %   cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
        %   bbci_cfy= [TODAY_DIR '/Lap_C3z4_bp2_' CLSTAG];
        %   cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' bbci_cfy ''');'];
        %   system(['matlab -nosplash -r "' cmd_init 'setup_season11; ' cmd_bbci '; exit &']);
        %   %system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);

        %pause(15) % WHY???

        ci1= find(CLSTAG(1)=='LRF');
        ci2= find(CLSTAG(2)=='LRF');
        classes= all_classes([ci1 ci2]);
        settings_bbci= {'start_marker', 210, ...
            'quit_marker', 254, ...
            'feedback', '1d', ... % This is not the name of the feedback! Just for postprocessing (Guido)
            'fb_port', 12345, ...
            'adaptation.policy', 'pcovmean', ...
            'adaptation.adaptation_ival', [550 1600], ...  % Vgl mit bbci.setup_opts.ival
            'adaptation.tmpfile', [tmpfile '_' CLSTAG], ...
            'adaptation.mrk_start', {ci1, ci2}, ...
            'adaptation.load_tmp_classifier', 1,...
            'adaptation.UC_mean', 0.075,...
            'adaptation.UC_pcov', 0.03}; % CARMEN
        %                  'adaptation.adaptation_ival', [750 4500], ...  % Vgl mit bbci.setup_opts.ival
        %'adaptation.load_tmp_classifier', ri>1,...
        settings_fb= {'classes', classes, ...
            'trigger_classes_list', {'left','right','foot'},  ...
            'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)], ...
            'trials_per_run', n_trials_per_run , ...
            'break_every', n_trials_per_run,...
            'cursor_visible',0 ...
            'duration_blank', 3000}; % break between trials

        %  fprintf('Press <RETURN> when ready to start Run %d�%d: classes %s and wait (press <RETURN only once!).\n', ri, ti, CLSTAG);
        %  pause; fprintf('Ok, starting ...\n');
        fprintf('Starting Run %d %d: classes %s.\n', ceil(ri/2), ti+3*mod(ri-1,2), CLSTAG);
        if ri+ti>2,
            pause(5);  %% Give time to display class combination, e.g. 'L vs R'
        end
        %nogui_start_feedback(setup, 'impedances',ri+ti==2);
%         nogui_start_feedback(setup, 'impedances',0);
        %nogui_start_feedback(setup, 'impedances',1);
        
        display('starting pyff')
        PYFF_DIR = 'E:\svn\pyff\src';
        general_port_fields= struct('bvmachine','127.0.0.1',...
            'control',{{'127.0.0.1',12471,12487}},...
            'graphic',{{'',12487}});
        general_port_fields.feedback_receiver= 'pyff';
        pyff('startup', 'gui',1, 'dir',PYFF_DIR);
        pyff('init','FeedbackCursorArrow')

        pause(2)

        fb_opt= []; fb_opt_int= []; % why clear the settings ???

        fb_opt.availableDirections= {'right','left'};
        fb_opt_int.pauseAfter = 20;
        fb_opt_int.hitMissDuration =  2000;
        fb_opt_int.trials = 50;

        pyff('set', fb_opt);
        pyff('setint', fb_opt_int);
        pyff('play')

        fprintf('type dbcont when feedback has finished (windows must have closed).\n');
        keyboard; 
        fprintf('Thank you for letting me know ...\n');
    end
end

%% - Train 'CSP_24chans' on Feedbacks Runs
bbci= bbci_default;
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_LapC3z4_*');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_24chans');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1, 'store_invcov',1};
bbci.setup_opts.clab= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
% only for testing (thomas)
%bbci.impedance_threshold=inf;
%warning(' impedance threshold set to inf for testing')

bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci_bet_finish


%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
nRuns=1;

setup_file= 'season11\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_24chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');

% Carmen
%attention I suppose feedback duration 3000 mseconds and period between cue and feedback S.bbci.analyze.ival(1),
%but the first time, this period is 1000 ms, as no data from the subject is available.

adaptation_ival=S.bbci.analyze.ival;
if adaptation_ival(1)<250, 
    adaptation_ival(1)=250;
end;
if adaptation_ival(2)<2750
    adaptation_ival(2)=2750;
end
if adaptation_ival(2)>5000;
    adaptation_ival(2)=5000;
end;
old_ival=adaptation_ival;
%fprintf('Press <RETURN> when ready to start Run and wait (press <RETURN only once!).\n');
%desc= stimutil_readDescription(['season11_imag_fbarrow_' CLSTAG]);
%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
for i =1:nRuns
  system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(7)
  settings_bbci= {'start_marker', 210, ...
    'quit_marker', 254, ...
    'feedback', '1d', ...
    'fb_port', 12345, ...
    'adaptation.mrk_start', [ci1 ci2], ...
    'adaptation.load_tmp_classifier', i>1,...
    'adaptation.UC',0.05, ...
    'adaptation.adaptation_ival', adaptation_ival}; % CARMEN
  settings_fb= {'classes', classes, ...
    'trigger_classes_list', {'left','right','foot'},  ...
    'duration_before_free', adaptation_ival(1), ...
    'trials_per_run', 30, ... 
    'break_every', 30,...
    'cursor_visible',0, ... % 1: cursor always visible, 0: cursor only visible at the end
    'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)],...
    'duration_blank', 3000}; % break between trials
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  nogui_send_setup(setup);
%   pause;
  nogui_start_feedback(setup, 'impedances',0);
  fprintf('When feedback has finished, press <RETURN> to continue.\n');
  pause; fprintf('Thank you for letting me know ...\n');

  %Cursor-Off Run:
  settings_fb= {'classes', classes, ...
    'cursor_on', 1, ...  % plus weitere Parameter:
    'cursor_visible',1,... % switch for cursor in active state only
    'response_at', 'center', ...
    'trigger_classes_list', {'left','right','foot'},  ...
    'duration_before_free', S.bbci.analyze.ival(1), ...
    'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};

end
%% - Train 'CSP_48chans' on Feedbacks Runs
% Check number of channels
bbci_tmp_class = bbci.classes;
bbci= bbci_default;
bbci.classes = bbci_tmp_class;
bbci.train_file= strcat(bbci.subdir, {'/imag_fbarrow_pmean*'});
bbci.train_file= strcat(bbci.subdir, {'/imag_fbarrow_pmean*', '/imag_fbarrow_LapC3z4_*'});
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1, 'store_invcov',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};

% only for testing (thomas)
%bbci.impedance_threshold=inf;
%warning(' impedance threshold set to inf for testing')

bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci_bet_finish


%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation, answer

%% questions
setup_file= 'season11\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.graphic_player1.feedback_opt.type= 'feedback_cursor_arrow_questions'; % 
setup.general.savestring= 'imag_fbarrow_dialog_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_48chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
clear S;
S= load(bbci_cfy);
classes= S.bbci.classes;
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
  
questions={'Sind Sie in Berlin geboren?','ja','nein';...
           'Haben Sie auch einen Sohn?','ja','nein';...
           'Ist der Hase weiblich?','ja','nein';...
           'Sind Sie verheiratet?','ja','nein';...
           'Haben Sie Klavier gespielt?','ja','nein';...
           'Ueber der Tuer haengen Hufeisen - sind Sie geritten?','ja','nein';...
           'Werden Sie sich den Film THIS IS IT �ber Jacko ansehen?','ja','nein';...
           'Sollen wir fuer Sie ein Photo von diesem Experiment machen?','ja','nein';...
           'Wuerden Sie nochmals an einem aehnlichen Experiment teilnehmen?','ja','nein'}  ;

adaptation_ival=S.bbci.analyze.ival;
if adaptation_ival(1)<250, 
    adaptation_ival(1)=250;
end;
if adaptation_ival(2)<2750
    adaptation_ival(2)=2750;
end
if adaptation_ival(2)>5000;
    adaptation_ival(2)=5000;
end;
old_ival=adaptation_ival;
             
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(10)
  settings_bbci= {'start_marker', 210, ...
                  'quit_marker', 254, ...
                  'feedback', '1d', ...
                  'fb_port', 12345, ...
                  'adaptation.mrk_start', [ci1 ci2], ...
                  'adaptation.adaptation_ival',adaptation_ival,...
                  'adaptation.load_tmp_classifier', 0,...
                  'adaptation.UC',0.05}; % CARMEN
  settings_fb= {'classes', classes, ...
                'trigger_classes_list', {'left','right','foot'},  ...
                'duration_before_free', adaptation_ival(1), ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)],...
                'cross_visible',0,... % turn the cursor on and off
                'questions',questions};
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  nogui_send_setup(setup);
  fprintf('Press <RETURN> when ready to start Run and wait (press <RETURN only once!).\n');
  desc= stimutil_readDescription(['season11_imag_fbarrow_' CLSTAG]);
  stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
  nogui_start_feedback(setup, 'impedances',0);
  fprintf('When feedback has finished, press <RETURN> to continue.\n');
  pause; fprintf('Thank you for letting me know ...\n');

 
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season11_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the FEEDBACK runs.\n');
pause
close all

