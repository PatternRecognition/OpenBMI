%-newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause

%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season12_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the FEEDBACK runs.\n');
pause
close all

%-newblock
%% - Runs 1, 30 trials per class, with pcovmean and subject independent classifier on c3,z,4
desc= stimutil_readDescription(['season12_imag_fbarrow_' CLSTAG]); % TODO create in acquisition/data/task_descriptions/
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

bbci_cfy = bbci_apply_loadSettings([TODAY_DIR cfy_name]);
bbci_cfy.adaptation= adaptation_default;
%% TODO: Change to best values
bbci_cfy.adaptation.UC_mean= 0.075;
bbci_cfy.adaptation.UC_pcov= 0.03;
bbci_cfy.adaptation.offset= 750;
%% TODO: Ask, maybe not needed all 3 below
bbci_cfy.adaptation.tmpfile = [TODAY_DIR 'adaptation_pcovmean_cspp_' today_str];
bbci_cfy.adaptation.mrk_start= [1 2];
bbci_cfy.adaptation.load_tmp_classifier= 0;

bbci_cfy.quit_condition.marker= 254;

setup_file= 'season12\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
settings_fb= {'classes', bbci.classes, ...
  'trigger_classes_list', bbci.classes,  ...
  'pause_msg', 'relax', ...
  'trials_per_run', 60, ...
  'break_every', 30};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
bbci_cfy.feedback.param= setup.graphic_player1.feedback_opt;

bvr_startrecording(['imag_fbarrow_cspp_C3z4_' PATCH '_pcovmean']);
fprintf('Starting Run 1, 60 Trials.\n');
bbci_apply(bbci_cfy)
bvr_stoprecording;


%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season12_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);

%-newblock
%% - Train 'CSP + 6 Patches' on Feedback Run 1
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_C3z4_*');
bbci.setup= 'cspp_auto';

%% TODO: is the field setup_opts needed for run1? Shall I remove it and set it again?
bbci.setup_opts= [];
bbci.setup_opts.classes= classes;
bbci.setup_opts.model= {'RLDAshrink', 'scaling', 1};
%% TODO: check PO1,z,2. If PO3,z,4 is on the cap than change the getLaplacian with
%% twelve accordingly, i.e. a special case for the patch in Pz
bbci.setup_opts.clab= {'F5-6','FFC5-6','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6', ...
  'PCP5-6','P5-6','PPO1,2','PO1,z,2'};
%% 48 channels for csp at first
bbci.setup_opts.clab_csp= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO1,z,2'};
bbci.setup_opts.band= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.patch= 'auto';

%% TODO: save_name is directly in bbci? Is the same of bbci.adaptation.filename?
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_csppauto_csp48chans_run1');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish

%% adaptation
bbci.adaptation.fcn= @bbci_adaptation_cspp_auto;
%% TODO: ival default or optimal = bbci.setup_opts.ival? Or adapt just the beginning of adaptation?
bbci.adaptation.param= {struct('ival',[500 4000])};
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_csp48chans_retrain';

%% -newblock
%% - Run 2:
%% - BBCI adaptive Feedback, classifier from above, adaptation by retraining
setup_file= 'season12\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_csppauto_csp48chans_run1'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
S= load(bbci_cfy);
settings_bbci= {'start_marker', 210, ...
  'quit_marker', 254, ...
  'feedback', '1d', ...
  'fb_port', 12345, ...
  'adaptation.policy', 'cspp', ...
  'adaptation.running', 1, ...
  'adaptation.offset',  S.bbci.analyze.ival(1), ...
  'adaptation.mrk_start', {1, 2}, ...
  'adaptation.load_tmp_classifier', 0};
settings_fb= {'classes', classes, ...
  'trigger_classes_list', classes,  ...
  'duration_before_free', S.bbci.analyze.ival(1), ...
  'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
%% TODO: what shall I do with the offset?
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
setup.general.savestring= 'imag_fbarrow_cspp_retrain_run1';
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start Run 2 and wait (press <RETURN only once!).\n');
pause(5);
desc= stimutil_readDescription(['season12_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);
fprintf('Press <RETURN> when feedback has finished to continue.\n');
pause; fprintf('Thank you for letting me know ...\n');

%% - Train 'CSP + 6 Patches' on Feedback Run 1 and 2
%% TODO: maybe hold bbci and not put it as default, but just put to auto what we want to calculate again. I
%% commented everything I think can be deleted
% bbci= bbci_default;
% bbci.classes= classes;
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_*');
% bbci.setup= 'cspp_auto';
% bbci.setup_opts= [];
% bbci.setup_opts.classes= classes;
% bbci.setup_opts.model= {'RLDAshrink', 'scaling', 1};
%% TODO: check PO1,z,2. If PO3,z,4 is on the cap than change the getLaplacian with
%% twelve accordingly, i.e. a special case for the patch in Pz
% bbci.setup_opts.clab= {'F5-6','FFC5-6','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6', ...
%   'PCP5-6','P5-6','PPO1,2','PO1,z,2'};
%% TODO: Everything auto or hold the band and the ival?
bbci.setup_opts.clab_csp= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO1,z,2'};
bbci.setup_opts.band= 'auto';
bbci.setup_opts.ival= 'auto';
% bbci.setup_opts.patch= 'auto';

bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_csppauto_run12');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish

%% TODO: ival default or optimal = bbci.setup_opts.ival? Or adapt just the beginning of adaptation?
bbci.adaptation.param= {struct('ival',[500 4000])};
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_retrain';


%% -newblock
%% - Run 3:
%% - BBCI adaptive Feedback, classifier from above, adaptation by retraining
setup_file= 'season12\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_csppauto_run12'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
S= load(bbci_cfy);
settings_bbci= {'start_marker', 210, ...
  'quit_marker', 254, ...
  'feedback', '1d', ...
  'fb_port', 12345, ...
  'adaptation.policy', 'cspp', ...
  'adaptation.running', 1, ...
  'adaptation.offset',  S.bbci.analyze.ival(1), ...
  'adaptation.mrk_start', {1, 2}, ...
  'adaptation.load_tmp_classifier', 0};
settings_fb= {'classes', classes, ...
  'trigger_classes_list', classes,  ...
  'duration_before_free', S.bbci.analyze.ival(1), ...
  'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
%% TODO: what shall I do with the offset?
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
setup.general.savestring= 'imag_fbarrow_cspp_retrain_run2';
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start Run 3 and wait (press <RETURN only once!).\n');
pause(5);
desc= stimutil_readDescription(['season12_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup);
fprintf('Press <RETURN> when feedback has finished to continue.\n');
pause; fprintf('Thank you for letting me know ...\n');

%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season12_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
close all

%-newblock
%% - Train CSP-based classifier on Feedback Runs 2, 3
%% TODO: maybe hold bbci and not put it as default, but just put to auto what we want to calculate again
bbci= bbci_default;
%% TODO: use run 2-3 or also run 1?
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_retrain_*');
bbci.setup= 'cspp_auto';
%% TODO: now still csp with 48 channels? Everything auto or hold the band and the ival? csp+patches or just patches?
% bbci.setup_opts.clab_csp= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO1,z,2'};
bbci.setup_opts.clab_csp= bbci.setup_opts.clab;
% bbci.setup_opts.band= 'auto';
% bbci.setup_opts.ival= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};

bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_csppauto_run23');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

%% Adaptation
bbci.adaptation.fcn= @bbci_adaptation_pmean;
%% TODO: ival default or optimal = bbci.setup_opts.ival? Or adapt just the beginning of adaptation?
bbci.adaptation.param= {struct('ival',[500 4000])};
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_pmean';
%% TODO: put the best UC value
bbci.adaptation.UC= 0.05;

%% change cont_proc and feature, the filters stay the same
[filt_b, filt_a]= butter(5, band/fs_eeg*2);
bbci.cont_proc.clab = fv.origClab;
bbci.cont_proc.proc= {{@online_linearDerivation, bbci.analyze.spat_w, bbci.analyze.features.clab}, {@online_filt, bbci.setup_opts.filt_b, bbci.setup_opts.filt_a}};

%% Feature extraction - feature. TODO: 750 or 500?
bbci.feature.proc= {@proc_variance, @proc_logarithm};
bbci.feature.ival= [-750 0];
bbci_bet_finish


%-newblock
%% TODO: how many runs?
% - Runs 4
% - BBCI adaptive Feedback, CSPP-based classifier, pmean adaptation
setup_file= 'season12\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_csppauto_run23'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
for ri= 1:2,  
  system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(7)
  settings_bbci= {'start_marker', 210, ...
                  'quit_marker', 254, ...
                  'feedback', '1d', ...
                  'fb_port', 12345, ...
                  'adaptation.mrk_start', [1 2], ...
                  'adaptation.load_tmp_classifier', ri>1};
  settings_fb= {'classes', classes, ...
                'trigger_classes_list', classes,  ...
                'duration_before_free', S.bbci.analyze.ival(1), ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  nogui_send_setup(setup);
  fprintf('Press <RETURN> when ready to start Run 4 and wait (press <RETURN only once!).\n');
  desc= stimutil_readDescription(['season12_imag_fbarrow_' CLSTAG]);
  stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
  nogui_start_feedback(setup);
  pause;
end

%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season12_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the next FEEDBACK run.\n');
pause
close all

%% TODO: don't remember what was this!
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
