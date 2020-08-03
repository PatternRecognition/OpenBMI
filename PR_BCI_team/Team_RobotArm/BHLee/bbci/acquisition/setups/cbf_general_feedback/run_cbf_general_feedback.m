%% Preparation
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause

%% Calibration by Feedback
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15;16-35] at Lap C3,z,4), pcovmean adaptation
setup_file= 'season10\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
tag_list= {'LR', 'LF', 'FR'};
all_classes= {'left', 'right', 'foot'};

cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/Lap_C3z4_bp2_' CLSTAG];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
classes= all_classes([ci1 ci2]);
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.policy', 'pcovmean', ...
                'adaptation.offset', 750, ...
                'adaptation.mrk_start', {ci1, ci2}};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'trials_per_run', 100};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
setup.general.savestring= ['imag_fbarrow_LapC3z4_' CLSTAG];
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start: classes %s and wait (press <RETURN only once!).\n', CLSTAG);
pause; fprintf('Ok, starting ...\n');
nogui_start_feedback(setup);



%% Train CSP-based Classifier on Calibration data
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.train_file= strcat(bbci.subdir, '/imag_arrow*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3,4','FC5-6','C5-6','CCP3,4','CP3,z,4','P5,1,2,6'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_csp_prelim');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci.adaptation.offset= bbci.setup_opts.ival(1)+250;
bbci.adaptation.UC= 0.05;
bbci_bet_finish


%% - Feedback Run 1 with preliminary CSP classifier
%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
% Load a setup file, that was used with Matlab GUI
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
% File name for the EEG file
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_csp_prelim'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.mrk_start', [ci1 ci2]};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', 1000, ...
              'trials_per_run', 100, ...
              'break_every', 25};
%              'duration_before_free', S.bbci.analyze.ival(1), ...
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start (press <RETURN only once!).\n');
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);


%% Train CSP-based Classifier on Calibration + 1. Run of Feedback data
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.train_file= strcat(bbci.subdir, '/imag_*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_csp');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci.adaptation.offset= bbci.setup_opts.ival(1)+250;
bbci_bet_finish


%% - Feedback Run 2 with final CSP classifier
%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_csp'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.mrk_start', [ci1 ci2]};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', 1000, ...
              'trials_per_run', 100, ...
              'break_every', 25};
%              'duration_before_free', S.bbci.analyze.ival(1), ...
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start (press <RETURN only once!).\n');
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);




%% - Feedback Run 3 with final CSP classifier
%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_csp'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.mrk_start', [ci1 ci2]};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', 1000, ...
              'trials_per_run', 100, ...
              'break_every', 25};
%              'duration_before_free', S.bbci.analyze.ival(1), ...
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start (press <RETURN only once!).\n');
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);
