%% - Motor Imagery Calibration: Left vs. Right
stim= [];
stim.cue= struct('classes', {'left','right'});
[stim.cue.nEvents]= deal(50);
[stim.cue.timing]= deal([0 4000 2000]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.handle_background= stimutil_initFigure;
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

fprintf('Press <RETURN> when ready to start ''imagined movement'' measurement.\n');
desc= stimutil_readDescription('vitalbci_season1_imag_arrow');
h_desc= stimutil_showDescription(desc, 'waitfor','key');

opt.filename= 'imag_arrow';
opt.breaks= [25 15];  %% Alle 25 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
stim_visualCues(stim, opt);

%% - Training of Initial Classifier
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.train_file= strcat(bbci.subdir, '/imag_arrowVPzk');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_24chans');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci.adaptation.UC= 0.05;
bbci_bet_finish


%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_24chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.mrk_start', [ci1 ci2], ...
                'adaptation.load_tmp_classifier', 0};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', S.bbci.analyze.ival(1), ...
              'trials_per_run', 100, ...
              'break_every', 25};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start (press <RETURN only once!).\n');
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup, 'impedances',0);

%% - Training of Final Classifier
bbci= bbci_default;
bbci.setup= 'cspauto';
%bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow*'); % only Fb data?
bbci.train_file= strcat(bbci.subdir, '/imag_*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci_bet_prepare
%bbci.setup_opts.reject_opts= {'whiskerperc',5};
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
bbci.adaptation.running= 0;
bbci.adaptation.policy= 'pmean';
bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci.adaptation.UC= 0.05;
bbci_bet_finish

%% - Prepare Pinball Feedback
fprintf('Plug in flipper trigger cable.\n');

bbci_setup= bbci.save_name;

%% Start Matlab GUI in Matlab 2 and Start Online Classifier in Matlab 1
%  These lines have to be executed without delay between them!
system(['matlab -r "dbstop if error; matlab_control_gui(''flipper_hardware'', ''classifier'',''' bbci_setup ''');" &']);
bbci_bet_apply(bbci_setup, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);
