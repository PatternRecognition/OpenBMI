%% newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause

%% newblock - Relax measurement
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season13_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the FEEDBACK runs.\n');
pause
close all

% Adaptation
uc = 2.^(-[10:5:80]./8);
switch VP_CAT 
  case 1
    iUC_mean= 7;
    iUC_pcov= 8;
  case 2
    iUC_mean= 6;
    iUC_pcov= 7;
  case 3
    iUC_mean= 5;
    iUC_pcov= 6;
end

%% newblock - Runs 1, 40 trials per class, with pcovmean and subject independent classifier on c3,z,4
if ischar(CLSTAG)
  desc= stimutil_readDescription(['season13_imag_fbarrow_' CLSTAG '_run1']);  
  tag_list= {CLSTAG};
  fb.classes= bbci.classes;
else
  desc= stimutil_readDescription(['season13_imag_fbarrow_LRF']);
  tag_list= {'LR', 'LF', 'FR'};
  fb.classes= {'left','right'};
end

stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

% %% pyff
fb.countdownFrom= int16(5);
fb.classesMarkers= [int16(1) int16(2)];
fb.classesDirections= bbci.classDef(2,:);
fb.trialsPerClass = int16(2);
fb.PauseAfter= 4;

pyff('init','FeedbackCursorArrow3');
pause(4)

pyff('set', fb);

%% run
pause(5);
pyff('play')

fprintf('Press <RETURN> when the test run is finished')
pause
pyff('stop');
pyff('quit');
  
fb.countdownFrom= int16(15);
fb.PauseAfter= int16(16);

for ti= 1:length(tag_list)
  
  % TODO: not necessary, change the way to save the variables in the subject
  % independent classifier so that it works with bbci_apply_loadSettings
  % bbci_cfy = bbci_apply_loadSettings([TODAY_DIR cfy_name]);
  cfy_name= ['patches_C3z4_small_8-32_' tag_list{ti}];
  % For experiment with naive people, we start the first run with 3 classes and we hold
  % the marker 3 for foot also in the following runs with 2 classes. With
  % people with already known classes, we use always markers 1 and 2
  if length(tag_list) == 1
    clidx1= 1;
    clidx2= 2;
    classes = bbci.classes;
  else
    clidx1= find(tag_list{ti}(1)=='LRF');
    clidx2= find(tag_list{ti}(2)=='LRF');
    classes= all_classes([clidx1 clidx2]);
  end
  
  bbci_cfy= load([TODAY_DIR cfy_name]);
  bbci_cfy= merge_structs(bbci_cfy, bbci_default);
  bbci_cfy.adaptation= bbci_default.adaptation;
  bbci_cfy.adaptation.fcn= @bbci_adaptation_pcovmean;
  bbci_cfy.adaptation.param= {struct('ival',[750 3750], 'UC_mean', uc(iUC_mean),'UC_pcov', uc(iUC_pcov),'mrk_start', [clidx1 clidx2])};
  bbci_cfy.adaptation.filename= [TODAY_DIR 'bbci_classifier_cspp_C3z4_' patch '_' bandstr '_' tag_list{ti} '_pcovmean'];
  bbci_cfy.log.filebase= 'log';
  % FBACK parameter to change
  fb.classes= classes;
  fb.classesDirections= bbci_cfy.classDef(2,:);  
  fb.classesMarkers= [int16(clidx1) int16(clidx2)];
  fb.trialsPerClass = int16(40);

  pyff('init','FeedbackCursorArrow3');
  pause(3)

  pyff('set', fb);
  pyff('save_settings', [pyff_fb_setup '_pcovmean']);

  %% run
  pause(2);
  pyff('play','basename', ['imag_fbarrow_cspp_C3z4_pcovmean_' tag_list{ti}]);
  bbci_apply(bbci_cfy);

  pause(1);
  bbci_acquire_bv('close');
  pyff('stop');
  fprintf('close the pyff console');
  pyff('quit');
  keyboard

end
  
%% newblock, Run 2 - Train 'CSP24channels + 6 Patches' on Feedback Run 1
bbci= bbci_default;
bbci.setup= 'cspp_auto';
bbci.setup_opts= [];
if length(tag_list) == 1
  bbci.train_file{ti}= strcat(bbci.subdir, '/imag_fbarrow_cspp_C3z4_pcovmean*');
else
  for ti= 1:length(tag_list)
    bbci.setup_opts.events{ti} = (ti-1)*80+1:ti*80;
    bbci.train_file{ti}= strcat(bbci.subdir, ['/imag_fbarrow_cspp_C3z4_pcovmean_' tag_list{ti} '*']);
  end
end
bbci.setup_opts.classes= bbci.classes;
bbci.setup_opts.model= {'RLDAshrink', 'scaling', 1};
bbci.setup_opts.clab_csp= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci.setup_opts.band= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.patch= 'auto';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp24_run1');
bbci_bet_prepare
bbci_bet_analyze  
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

% class chosen, CLSTAG can be set, if not yet set
if length(tag_list) == 3
  CLSTAG= [upper(bbci.classes{1}(1)) upper(bbci.classes{2}(1))];
  clidx1= find(CLSTAG(1)=='LRF');
  clidx2= find(CLSTAG(2)=='LRF');
end

%% Adaptation
bbci.adaptation.fcn= @bbci_adaptation_cspp;
bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'featbuffer', fv, 'mrk_start', [clidx1 clidx2])};
clear fv
bbci.adaptation.filename= [TODAY_DIR 'bbci_classifier_cspp_csp24_retrain_run1'];
bbci.adaptation.mode = 'everything_at_once';
bbci.log.filebase= [bbci.save_name '_log/log'];
bbci_bet_finish;

% in retrain the cspp change
bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};

% FBACK parameter to change
fb.classes = bbci.classes;
fb.classesDirections= bbci.classDef(2,:);  
fb.classesMarkers= [int16(clidx1) int16(clidx2)];
fb.trialsPerClass = int16(50);

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('save_settings', [pyff_fb_setup '_retrain']);

% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_csp24_retrain');
bbci_apply(bbci);

pause(1);
pyff('stop');
pyff('quit'); 

%% newblock - Run 3 - BBCI adaptive Feedback, classifier from above, adaptation by retrain
% Train 'CSP48channels + 6 Patches' on Feedback Run 1 and 2
if length(tag_list) == 1
  bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_*');
else
  bbci.setup_opts= rmfield(bbci.setup_opts,'events');
  bbci.train_file=  strcat(bbci.subdir, {'/imag_fbarrow_cspp_csp24_retrain*',['/imag_fbarrow_cspp_C3z4_pcovmean_' CLSTAG '*']});
end
bbci.setup_opts.clab_csp= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.setup_opts.ival= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
% bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp48_run12');
fprintf('Decide whether to set band and patch again to auto');
keyboard
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

%% Adaptation
bbci.adaptation.fcn= @bbci_adaptation_cspp;
bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'featbuffer', fv, 'marker_start', [clidx1 clidx2])};
bbci.adaptation.filename= [TODAY_DIR 'bbci_classifier_cspp_csp48_retrain_run2'];
bbci.adaptation.mode = 'everything_at_once';

bbci.log.filebase= [bbci.save_name '_log/log'];

bbci_bet_finish;

% in retrain the cspp change
bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('save_settings', [pyff_fb_setup '_retrain']);

%% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_csp48_retrain');
bbci_apply(bbci);

pause(1);
pyff('stop');
pyff('quit'); 
keyboard;

%% newblock - Relax measurement
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season13_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
close all

%% newblock - Run 4 - BBCI adaptive Feedback, classifier from above, adaptation by retrain
% Train 'CSP48channels + 6 Patches' on Feedback Run 2 and 3
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_csp*');
bbci.setup_opts.ival= 'auto';
% bbci.setup_opts.patch= 'auto';
% bbci.setup_opts.band= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
bbci.setup_opts.nPat_csp= 3;
bbci.setup_opts.nPat= 3;
bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp48_run23');
fprintf('Decide whether to set band and patch again to auto');
keyboard
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

%% Adaptation
bbci.adaptation.fcn= @bbci_adaptation_cspp;
bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'featbuffer', fv, 'mrk_start', [clidx1 clidx2])};
bbci.adaptation.filename= [TODAY_DIR 'bbci_classifier_cspp_csp48_retrain_run3'];
bbci.adaptation.mode = 'everything_at_once';

bbci.log.filebase= [bbci.save_name '_log/log'];

bbci_bet_finish;

% in retrain the cspp change
bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('save_settings', [pyff_fb_setup '_retrain']);
%% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_csp48_retrain');
bbci_apply(bbci);

pause(1);
pyff('stop');
pyff('quit'); 
keyboard;

%% newblock - Train CSP-based classifier on Feedback Runs 2, 3 and 4
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_csp*');
bbci.setup_opts.ival= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.setup_opts.model= {'RLDAshrink', 'gamma', 0, 'scaling', 1, 'store_means', 1};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp48_run234');
fprintf('Decide whether to set band and patch again to auto');
keyboard
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

%% Adaptation
switch VP_CAT 
  case 1
    iuc= 9;
  case 2
    iuc= 7;
  case 3
    iuc= 5;
end
bbci.adaptation.fcn= @bbci_adaptation_pmean;
bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'UC', uc(iuc), 'mrk_start', [clidx1 clidx2])};
bbci.adaptation.filename= [TODAY_DIR 'bbci_classifier_cspp_csp48_pmean'];
bbci.adaptation.mode= 'classifier';

bbci.log.filebase= [bbci.save_name '_log/log'];
bbci_bet_finish

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('save_settings', [pyff_fb_setup '_pmean']);

%% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_pmean');
bbci_apply(bbci);

pause(1);
bbci_acquire_bv('close');
pyff('stop');
pyff('quit');

keyboard;

%% newblock - Relax measurement
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season13_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
close all

%% newblock - Train CSPP for subsequent session
bbci.train_file= {[bbci.subdir, '/imag_fbarrow_cspp_csp*'],[bbci.subdir, '/imag_fbarrow_cspp_pmean*']};
bbci.setup_opts.ival= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.setup_opts.model= {'RLDAshrink', 'gamma', 0, 'scaling', 1, 'store_means', 1, 'store_invcov',1};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_carryover_run2345');
fprintf('Decide whether to set band and patch again to auto');
keyboard
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.log.filebase= [bbci.save_name '_log/log'];
bbci_bet_finish