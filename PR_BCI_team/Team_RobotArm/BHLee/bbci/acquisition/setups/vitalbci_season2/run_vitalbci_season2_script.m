%% --- - --- Pretraining in three different groups

run_intervention;


%-newblock
%% --- - --- Relaxation 

[seq, wav, opt]= setup_vitalbci_season2_relax;
stimutil_waitForInput('msg_next','to start RELAX recording.');
stim_artifactMeasurement(seq, wav, opt);
clear seq wav opt


%-newblock
%% --- - --- Prepare Kickstart classifier for Runs 1a,b,c

bbci= load(kickstart_cfy);
bbci= copy_subfields(bbci, bbci_default);
bbci.source.record_basename= ['imag_fbarrow_kickstart' VP_CODE];
bbci.feedback= feedback_arrow_training;
[bbci.adaptation.log]= deal(struct('output','screen&file'));
bbci.calibrate.save.file= 'bbci_kickstart_classifier';
bbci_save(bbci);


%-newblock
%% --- - --- Run 1a: Kickstart classifier (3x binary, pcovmean adaptation)

desc= stimutil_readDescription('vitalbci_season2_cursor_training');
stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_waitForInput('msg_next','to start RUN 1a.');
bbci= load([TODAY_DIR 'bbci_kickstart_classifier']);
bbci.feedback.opt.seq_mod= 1;
data= bbci_apply(bbci);


%-newblock
%% --- - --- Run 1b: as above

stimutil_waitForInput('msg_next','to start RUN 1b.');
bbci= load([TODAY_DIR 'bbci_kickstart_classifier']);
bbci.feedback.opt.seq_mod= 2;
[bbci.adaptation.load_classifier]= deal(1);
data= bbci_apply(bbci);


%-newblock
%% --- - --- Run 1c: as above

stimutil_waitForInput('msg_next','to start RUN 1c.');
bbci= load([TODAY_DIR 'bbci_kickstart_classifier']);
bbci.feedback.opt.seq_mod= 3;
[bbci.adaptation.load_classifier]= deal(1);
data= bbci_apply(bbci);


%-newblock
%% --- - --- Calibrate BBCI System on Runs 1a,b,c for Runs 2,3

bbci= struct('calibrate', BC);
bbci.calibrate.file= ['imag_fbarrow_kickstart' VP_CODE '*'];
bbci.calibrate.fcn= @bbci_calibrate_csp_plus_lap;
bbci.calibrate.settings.clab_csp= ...
    {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
[bbci, data]= bbci_calibrate(bbci);

%%
set(cat(2, data.all_results.figure_handles), 'Visible','on');
bbci_save_otherPics(bbci,data);
%%
bbci= copy_subfields(bbci, bbci_default);
bbci.source.record_basename= ['imag_fbarrow_CSP24_plus_lap' VP_CODE];
bbci.feedback= feedback_arrow;
bbci.feedback.opt.classes= data.result.classes;
adapt_opt.mrk_start= {strmatch(data.result.classes{1}, classDef(2,:)), ...
                      strmatch(data.result.classes{2}, classDef(2,:))};
bbci.adaptation.param= {adapt_opt};

bbci_save(bbci, data);
close all


%-newblock
%% --- - --- Run 2: CSP (24 channels) plus selected Laplacians

clstag= cellfun(@(x)(upper(x(1))), bbci.feedback.opt.classes);
desc= stimutil_readDescription(['vitalbci_season2_cursor_feedback_' clstag]);
stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_waitForInput('msg_next','to start RUN 2.');
bbci= load([TODAY_DIR 'bbci_classifier_csp_plus_lap']);
data= bbci_apply(bbci);


%-newblock
%% --- - --- Run 3: CSP (24 channels) plus selected Laplacians (cont'ed)

stimutil_waitForInput('msg_next','to start RUN 3.');
bbci= load([TODAY_DIR 'bbci_classifier_csp_plus_lap']);
bbci.adaptation.load_classifier= 1;
data= bbci_apply(bbci);


%-newblock
%% --- - --- Calibrate BBCI System on Runs 2,3 for Runs 4,5

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


%-newblock
%% --- - --- Run 4: CSP (47 channels) mit PMEAN adaptation

stimutil_waitForInput('msg_next','to start RUN 4.');
bbci= load([TODAY_DIR 'bbci_classifier_csp']);
data= bbci_apply(bbci);


%-newblock
%% --- - --- Run 5: CSP (47 channels) mit PMEAN adaptation (cont'ed)

stimutil_waitForInput('msg_next','to start RUN 5.');
bbci= load([TODAY_DIR 'bbci_classifier_csp']);
bbci.adaptation.load_classifier= 1;
data= bbci_apply(bbci);
fprintf('* Acquisition done.\n');


%-newblock
%% --- - --- Psychological Questionnaires

stimutil_waitForInput('msg_next','to complete questionnaires.');

run_questionnaires;


%-newblock
%% --- - --- Session finished

acq_vpcounter(session_name, 'close');
fprintf('Session %s finished.\n', session_name);
