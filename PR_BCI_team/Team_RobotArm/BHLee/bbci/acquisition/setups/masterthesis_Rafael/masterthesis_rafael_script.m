%% Pretraining in three different groups
% Pretraining consists of either (1) reading a task, (2) 2nd-hand-training or 
% (3) listening to a relaxation tape; probably not necessary

% run_intervention;

%% Relaxation 
% This is the eyes-open-eyes-closed part in order to check whether the EEG
% is correctly set-up and to check how good the signal is by observing
% occipital alpha

% [seq, wav, opt]= setup_vitalbci_season2_relax;
% stimutil_waitForInput('msg_next','to start RELAX recording.');
% stim_artifactMeasurement(seq, wav, opt);
% clear seq wav opt
% 
% check_relex_recording

%% Prepare Kickstart classifier for Runs 1a,b,c
% The kickstart classifier is (I think) the universal classifier that is
% adapted to the subject trial-by-trial. Runs 1-3 are basically the same,
% the only difference in the code is bbci.feedback.opt.seq_mod= 1

bbci= load(kickstart_cfy);
bbci= copy_subfields(bbci, bbci_default);
bbci.source.record_basename= ['imag_fbarrow_kickstart' VP_CODE];
bbci.feedback= feedback_arrow_training;
[bbci.adaptation.log]= deal(struct('output','screen&file'));
bbci.calibrate.save.file= 'bbci_kickstart_classifier';
bbci_save(bbci);

%% Run 1a: Kickstart classifier (3x binary, pcovmean adaptation)

desc= stimutil_readDescription('masterthesis_rafael_cursor_training');
stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_waitForInput('msg_next','to start RUN 1a.');
bbci= load([TODAY_DIR 'bbci_kickstart_classifier']);
bbci.feedback.opt.seq_mod= 1;
data= bbci_apply(bbci);

%% Run 1b: as above

stimutil_waitForInput('msg_next','to start RUN 1b.');
bbci= load([TODAY_DIR 'bbci_kickstart_classifier']);
bbci.feedback.opt.seq_mod= 2;
[bbci.adaptation.load_classifier]= deal(1);
data= bbci_apply(bbci);

%% Run 1c: as above

stimutil_waitForInput('msg_next','to start RUN 1c.');
bbci= load([TODAY_DIR 'bbci_kickstart_classifier']);
bbci.feedback.opt.seq_mod= 3;
[bbci.adaptation.load_classifier]= deal(1);
data= bbci_apply(bbci);

%% Calibrate BBCI System on Runs 1a,b,c for Runs 2,3
% Here the data from runs 1-3 are used to calibrate a classifier, namely 
% bbci_calibrate_csp_plus_lap. Clab_csp setting might be important, or are
% the selected channels standard? And do we use bbci_calibrate_csp_plus_lap
% or without laplacian filtering?

bbci= struct('calibrate', BC);
bbci.calibrate.file= ['imag_fbarrow_kickstart' VP_CODE '*'];
bbci.calibrate.fcn= @bbci_calibrate_csp_plus_lap;
bbci.calibrate.settings.clab_csp= ...
    {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
[bbci, data]= bbci_calibrate(bbci);

%% 
% Calibration is (apparently) visualized and saved
set(cat(2, data.all_results.figure_handles), 'Visible','on');
bbci_save_otherPics(bbci,data);

%%
% Here the feedback probably has to be set to 'pyff' or multibrain-game. I
% think this is the part where from the three calibration classes (left, right, feet)
% the best two are chosen (happens maybe in bbci.feedback.opt.classes=
% data.result.classes). adapt_opt.mark_start could be important? And what
% is bbci.source.record_basename?
% Calibration is saved.

bbci= copy_subfields(bbci, bbci_default);
bbci.source.record_basename= ['imag_fbarrow_CSP24_plus_lap' VP_CODE];
bbci.feedback.receiver= 'pyff';
bbci.feedback.opt.classes= data.result.classes;
adapt_opt.mrk_start= {strmatch(data.result.classes{1}, classDef(2,:)), ...
                      strmatch(data.result.classes{2}, classDef(2,:))};
bbci.adaptation.param= {adapt_opt};

bbci_save(bbci, data);
close all

%% Run 2: CSP (24 channels) plus selected Laplacians
% What exactly is the difference between run 2 and 3? Or none at all?

clstag= cellfun(@(x)(upper(x(1))), bbci.feedback.opt.classes);
desc= stimutil_readDescription(['masterthesis_rafael_feedback_' clstag]);
stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_waitForInput('msg_next','to start RUN 2.');
bbci= load([TODAY_DIR 'bbci_classifier_csp_plus_lap']);
data= bbci_apply(bbci);

%% Run 3: CSP (24 channels) plus selected Laplacians (cont'ed)

stimutil_waitForInput('msg_next','to start RUN 3.');
bbci= load([TODAY_DIR 'bbci_classifier_csp_plus_lap']);
bbci.adaptation.load_classifier= 1;
data= bbci_apply(bbci);

%% Calibrate BBCI System on Runs 2,3 for Runs 4,5
% Again calibration on runs 2,3 for 4 and 5. What is bbci_adaptation_mean?
% This time apparently 47 channels instead of 24, here apparently no
% laplacian filtering

bbci= struct('calibrate', BC);
bbci.calibrate.file= ['imag_fbarrow_CSP24_plus_lap' VP_CODE '*'];
bbci.calibrate.fcn= @bbci_calibrate_csp;
bbci.calibrate.settings.clab= ...
    {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,4'};
[bbci, data]= bbci_calibrate(bbci);
bbci= copy_subfields(bbci, bbci_default);

bbci.source.record_basename= ['imag_fbarrow_CSP47_pmean' VP_CODE];
bbci.feedback= feedback_arrow;
bbci.feedback.opt.classes= data.result.classes;
bbci.adaptation.fcn= @bbci_adaptation_pmean;
adapt_opt.mrk_start= [1 2 3];
bbci.adaptation.param= {adapt_opt};

bbci_save(bbci, data);
close all

%% Run 4: CSP (47 channels) mit PMEAN adaptation

stimutil_waitForInput('msg_next','to start RUN 4.');
bbci= load([TODAY_DIR 'bbci_classifier_csp']);
data= bbci_apply(bbci);


%% Run 5: CSP (47 channels) mit PMEAN adaptation (cont'ed)

stimutil_waitForInput('msg_next','to start RUN 5.');
bbci= load([TODAY_DIR 'bbci_classifier_csp']);
bbci.adaptation.load_classifier= 1;
data= bbci_apply(bbci);
fprintf('* Acquisition done.\n');

%% Psychological Questionnaires
% We'll need something like this too

stimutil_waitForInput('msg_next','to complete questionnaires.');

run_questionnaires;

%% Session finished

acq_vpcounter(session_name, 'close');
fprintf('Session %s finished.\n', session_name);
