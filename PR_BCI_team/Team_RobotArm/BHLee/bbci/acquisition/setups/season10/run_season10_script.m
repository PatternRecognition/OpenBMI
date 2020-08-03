%-newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
fprintf('\nLet participant fill out first questionnaire.\n');
pause

%-newblock
%% - Artifact measurement: Test recording
fprintf('\n\nArtifact test run.\n');
%[seq, wav, opt]= setup_season10_artifacts_demo('clstag', 'LRF');
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT TEST measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1);


%-newblock
%% - Artifact measurement: recording
fprintf('\n\nArtifact recording.\n');
%[seq, wav, opt]= setup_season10_artifacts('clstag', 'LRF');
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the RELAX measurement.\n');
pause


%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the FEEDBACK runs.\n');
pause
close all


%-newblock
%% - Runs 1, 2
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15;16-35] at Lap C3,4), pcovmean adaptation
desc= stimutil_readDescription('season10_imag_fbarrow_LRF');
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

nRuns= 4;
setup_file= 'season10\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pcovmean_Lap_' today_str];
tag_list= {'LR', 'LF', 'FR'};
all_classes= {'left', 'right', 'foot'};
for ri= 1:nRuns,
 for ti= 1:length(tag_list),  
  CLSTAG= tag_list{ti};
  cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
  bbci_cfy= [TODAY_DIR '/Lap_C3z4_bp2_' CLSTAG];
  cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
 system(['matlab -nosplash -r "' cmd_init 'setup_season10; ' cmd_bbci '; exit &']);
%   system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
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
                  'adaptation.tmpfile', [tmpfile '_' CLSTAG], ...
                  'adaptation.mrk_start', {ci1, ci2}, ...
                  'adaptation.load_tmp_classifier', ri>1};
  settings_fb= {'classes', classes, ...
                'trigger_classes_list', {'left','right','foot'},  ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)], ...
                'trials_per_run', 10, ...
                'break_every', 10};
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  setup.general.savestring= ['imag_fbarrow_LapC3z4_' CLSTAG];
  nogui_send_setup(setup);
%  fprintf('Press <RETURN> when ready to start Run %d°%d: classes %s and wait (press <RETURN only once!).\n', ri, ti, CLSTAG);
%  pause; fprintf('Ok, starting ...\n');
  fprintf('Starting Run %d°%d: classes %s.\n', ceil(ri/2), ti+3*mod(ri-1,2), CLSTAG);
  if ri+ti>2,
    pause(5);  %% Give time to display class combination, e.g. 'L vs R'
  end
  nogui_start_feedback(setup, 'impedances',ri+ti==2);
  fprintf('Press <RETURN> when feedback has finished (windows must have closed).\n');
  pause; fprintf('Thank you for letting me know ...\n');
 end
end


fprintf('\nLet participant fill out questionnaire and press <RETURN>.\n');
pause


%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);


%% - ZHAND
fprintf('Instruct participant for ZHAND. Press <RETURN> when ready to train classifier.\n');
pause
close all


%-newblock
%% - Train 'CSP + 6 sel Lap' on Feedbacks Runs 1, 2
bbci= bbci_default;
bbci.setup= 'lapcsp';
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_LapC3z4_*');
bbci.setup_opts.model= {'RLDAshrink', 'scaling',1};
bbci.setup_opts.clab_csp= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_lapcsp_24chans_runs12');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
%bbci.adaptation.running= 1;
%bbci.adaptation.policy= 'lapcsp';
%bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci_bet_finish


%-newblock
%% - Run 3:
%% - BBCI adaptive Feedback, classifier from above, adaptation by retraining
setup_file= 'season10\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_lapcsp_24chans_runs12'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
S= load(bbci_cfy);
classes= S.bbci.classes;
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
clidx1= strmatch(classes{1}, {'left','right','foot'});
clidx2= strmatch(classes{2}, {'left','right','foot'});
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.policy', 'lapcsp', ...
                'adaptation.running', 1, ...
                'adaptation.offset',  S.bbci.analyze.ival(1), ...
                'adaptation.mrk_start', {clidx1, clidx2}, ...
                'adaptation.load_tmp_classifier', 0};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', S.bbci.analyze.ival(1), ...
              'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
setup.general.savestring= 'imag_fbarrow_lapcsp_run1';
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start Run 3 and wait (press <RETURN only once!).\n');
pause(5);
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);
fprintf('Press <RETURN> when feedback has finished to continue.\n');
pause; fprintf('Thank you for letting me know ...\n');


fprintf('\nLet participant fill out questionnaire and press <RETURN>.\n');
pause

%-newblock
%% Tactile Oddball experiment
setup_season10_oddballTactile_demo;
fprintf('Press <RETURN> when ready to start the Tactile Oddball TEST.\n');
pause

cont = 0;
while cont == 0
  stim_oddballTactile(10, opt, 'test',1, 'countdown',3);
  keyboard
end
setup_season10_oddballTactile;
fprintf('Press <RETURN> when ready to start the Tactile Oddball experiment.\n');
pause
stim_oddballTactile(N, opt);


%% - Train 'CSP + 6 sel Lap' on Feedbacks Runs (1, 2) % CLSTAG + 3
bbci= bbci_default;
bbci.classes= classes;
bbci.setup= 'lapcsp';
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
bbci.train_file= strcat(bbci.subdir, {'/imag_fbarrow_lapcsp*',['/imag_fbarrow_LapC3z4_' CLSTAG '*']});
bbci.setup_opts.model= {'RLDAshrink', 'scaling',1};
bbci.setup_opts.clab_csp= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_lapcsp_24chans_runs123');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
%bbci.adaptation.running= 1;
%bbci.adaptation.policy= 'lapcsp';
%bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci_bet_finish


%-newblock
%% - Run 4:
%% - BBCI adaptive Feedback, classifier from above, adaptation by retraining
setup_file= 'season10\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_lapcsp_24chans_runs123'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
S= load(bbci_cfy);
classes= S.bbci.classes;
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
clidx1= strmatch(classes{1}, {'left','right','foot'});
clidx2= strmatch(classes{2}, {'left','right','foot'});
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.policy', 'lapcsp', ...
                'adaptation.running', 1, ...
                'adaptation.offset',  S.bbci.analyze.ival(1), ...
                'adaptation.mrk_start', {clidx1, clidx2}, ...
                'adaptation.load_tmp_classifier', 0};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', S.bbci.analyze.ival(1), ...
              'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
setup.general.savestring= 'imag_fbarrow_lapcsp_run2';
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start Run 3 and wait (press <RETURN only once!).\n');
pause(5);
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);
fprintf('Press <RETURN> when feedback has finished to continue.\n');
pause; fprintf('Thank you for letting me know ...\n');


fprintf('\nLet participant fill out questionnaire and press <RETURN>.\n');
pause


%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Plug out tactor and press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Instruct participant for STROOP. Press ESC+F5 to start STROOP, then press F10. Press <RETURN> when ready to train classifier.\n');
pause
close all


%-newblock
%% - Train CSP-based classifier on Feedback Runs 3, 4
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_lapcsp*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
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


-newblock
% - Runs 5, 6, 7
% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_48chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
for ri= 3:3,  
  system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(7)
  settings_bbci= {'start_marker', 210, ...
                  'quit_marker', 254, ...
                  'feedback', '1d', ...
                  'fb_port', 12345, ...
                  'adaptation.mrk_start', [ci1 ci2], ...
                  'adaptation.load_tmp_classifier', ri>1};
  settings_fb= {'classes', classes, ...
                'trigger_classes_list', {'left','right','foot'},  ...
                'duration_before_free', S.bbci.analyze.ival(1), ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  nogui_send_setup(setup);
  fprintf('Press <RETURN> when ready to start Run %d and wait (press <RETURN only once!).\n', 4+ri);
  desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
  stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
  nogui_start_feedback(setup);
  fprintf('\nLet participant fill out questionnaire and press <RETURN>.\n');
  pause;
  switch(ri),
   case 1, 
    fprintf('When feedback has finished, press ESC+F5 to start ZHAND, then press F10. Participant performs ZHAND. After that press <RETURN>.\n');
    pause; fprintf('Thank you for letting me know ...\n');
    
   case 2,
    fprintf('When feedback has finished, press <RETURN> for Tactile Oddball experiment.\n');
    pause;
    setup_season10_oddballTactile_level2;
    fprintf('Plug in Tactor and press <RETURN> when ready to start the Tactile Oddball experiment.\n');
    pause; fprintf('Thank you for letting me know ...\n');
    stim_oddballTactile(N, opt);
    
   case 3,
    fprintf('When feedback has finished, press <RETURN> to continue.\n');
    pause; fprintf('Thank you for letting me know ...\n');
  end
  pause; fprintf('Thank you for letting me know ...\n');
end

%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the next FEEDBACK run.\n');
pause
close all


%-newblock
%% - Train sel lap classifier for subsequent session
bbci= bbci_default;
bbci.setup= 'sellap';
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_pmean*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1, 'store_invcov',1};
bbci.setup_opts.nlaps_per_area= 1;
bbci.setup_opts.clab= {'not','E*','Fp*'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_sellap_carryover');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish
