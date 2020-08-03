fprintf('WHAT ORDER?? Pre-Training - Prepare Cap - Relax Recording??\n');


%% --- - --- Pretraining in three different groups

switch(mod(VP_NUMBER, 3)),
 case 1,
  system('start wmplayer');
  stimutil_waitForInput('msg_next','to start PMR.');
  %[snd, fs]= wavread([DATA_DIR 'studies/vitalbci_season2/Entspannung_PMR_22min_Yvonne_11025Hz.wav']);
  %wavplay(snd, fs);
  mp3file= [DATA_DIR 'studies/vitalbci_season2/Entspannung_PMR_22min_Yvonne.mp3'];
  system(sprintf('start wmplayer "%s"', mp3file));
  stimutil_waitForInput('msg_next','when finished.');
 case 2,
  stimutil_waitForInput('msg_next','to start 2HAND training.');
  run_2HAND_training;
  stimutil_waitForInput('msg_next','when finished.');
 case 0,
  stimutil_waitForInput('msg_next','to start showing the PDF file.');
  pdffile= [DATA_DIR 'studies/vitalbci_season2/paper.pdf'];
  system(sprintf('start acrord32 /A "page=1&toolbar=0" "%s"', pdffile));
  stimutil_waitForInput('msg_next','when finished.');
end


%% --- - --- Relaxation 

[seq, wav, opt]= setup_vitalbci_season2_relax;
desc= stimutil_readDescription('vitalbci_season2_relaxation_recording');
stimutil_showDescription(desc, 'waitfor',0);
stimutil_waitForInput('msg_next','to start RELAX measurement.');
stim_artifactMeasurement(seq, wav, opt);
clear seq wav opt

bvr_sendcommand('viewsignals');


%% --- - --- Run 1: Kickstart classifiers (3x binary, pcovmean adaptation)

bbci= load(kickstart_cfy);
bbci= copy_subfields(bbci, bbci_default);
bbci.source.record_basename= ['imag_fbarrow_kickstart' VP_CODE];
bbci.feedback= feedback_arrow_training;
[bbci.adaptation.log]= deal(struct('output','screen&file'));

desc= stimutil_readDescription('vitalbci_season2_cursor_training');
stimutil_showDescription(desc, 'waitfor',0);

% Run 1a
stimutil_waitForInput('msg_next','to start RUN 1a.');
data= bbci_apply(bbci);

% Run 1b
stimutil_waitForInput('msg_next','to start RUN 1b.');
[bbci.adaptation.load_classifier]= deal(1);
data= bbci_apply(bbci);

% Run 1c
stimutil_waitForInput('msg_next','to start RUN 1c.');
data= bbci_apply(bbci);


%% --- - --- Run 2: CSP (24 channels) plus selected Laplacians

bbci= struct('calibrate', BC);
bbci.calibrate.file= ['imag_fbarrow_kickstart' VP_CODE '*'];
bbci.calibrate.fcn= @bbci_calibrate_csp_plus_lap;
bbci.calibrate.settings.clab_csp= ...
    {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
[bbci, data]= bbci_calibrate(bbci);
bbci= copy_subfields(bbci, bbci_default);

bbci.source.record_basename= ['imag_fbarrow_CSP24_plus_lap' VP_CODE];
bbci.feedback= feedback_arrow;
bbci.feedback.opt.classes= data.result.classes;
adapt_opt.mrk_start= {strmatch(data.result.classes{1}, classDef(2,:)), ...
                      strmatch(data.result.classes{2}, classDef(2,:))};
bbci.adaptation.param= {adapt_opt};

bbci_save(bbci, data);
close all

stimutil_waitForInput('msg_next','to start RUN 2.');
data= bbci_apply(bbci);


%% --- - --- Run 3: CSP (24 channels) plus selected Laplacians (cont'ed)

clstag= cellfun(@(x)(upper(x(1))), bbci.feedback.opt.classes);
desc= stimutil_readDescription(['vitalbci_season2_cursor_feedback_' clstag]);
stimutil_showDescription(desc, 'waitfor',0);
stimutil_waitForInput('msg_next','to start RUN 3.');
bbci.adaptation.load_classifier= 1;
data= bbci_apply(bbci);


%% --- - --- Run 4: CSP (47 channels) mit PMEAN adaptation

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

stimutil_waitForInput('msg_next','to start RUN 4.');
data= bbci_apply(bbci);


%% --- - --- Run 5: CSP (47 channels) mit PMEAN adaptation (cont'ed)

stimutil_waitForInput('msg_next','to start RUN 5.');
bbci.adaptation.load_classifier= 1;
data= bbci_apply(bbci);

acq_vpcounter(session_name, 'close');

fprintf('* Acquisition done. Proceed with psychological tests.\n');
