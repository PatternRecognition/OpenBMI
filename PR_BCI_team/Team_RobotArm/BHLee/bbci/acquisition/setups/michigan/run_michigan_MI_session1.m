%% newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause

% %% newblock - Relax measurement
% % TODO: to decide...
% fprintf('\n\nRelax recording.\n');
% [seq, wav, opt]= setup_season13_relax;
% fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
% pause
% stim_artifactMeasurement(seq, wav, opt);
% fprintf('Press <RETURN> when ready to go to the FEEDBACK runs.\n');
% pause
% close all

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
% TODO: write the explanation in english
desc= stimutil_readDescription('season13_imag_fbarrow_LRF');
tag_list= {'LR', 'LF', 'FR'};

stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

%% pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0, 'gui',0);
pause(4)

%% test modus to give an example
fb.classes= {'left','right'};
fb.classesDirections= bbci.classDef(2,:);
fb.classesMarkers= [int16(1) int16(2)];
fb.trialsPerClass= int16(6);
fb.countdownFrom= int16(6);
fb.pauseAfter= int16(6);
fb.shortPauseCountdownFrom= int16(6);

pyff('init','FeedbackCursorArrow3');
pause(3)

pyff('set', fb);
pyff('setint', fbint)

pause(2);
pyff('play')
pause(10);

pyff('stop');
pyff('quit');

fprintf('Press <RETURN> when ready to start the real FEEDBACK runs.\n');
keyboard

fb.trialsPerClass= int16(25);
fb.countdownFrom= int16(15);
fb.pauseAfter= int16(10);
fb.shortPauseCountdownFrom= int16(16);

for ti= 1:length(tag_list)
  
  cfy_name= ['patches_C3z4_small_8-32_' tag_list{ti}];
  clidx1= find(tag_list{ti}(1)=='LRF');
  clidx2= find(tag_list{ti}(2)=='LRF');
  classes= all_classes([clidx1 clidx2]);  
  
  bbci_cfy= load([TODAY_DIR cfy_name]);
  bbci_cfy= merge_structs(bbci_cfy, bbci_default);
  bbci_cfy.adaptation= bbci_default.adaptation;
  bbci_cfy.adaptation.fcn= @bbci_adaptation_pcovmean;
  bbci_cfy.adaptation.param= {struct('ival',[750 3750], 'UC_mean', uc(iUC_mean),'UC_pcov', uc(iUC_pcov),'mrk_start', [clidx1 clidx2])};
  bbci_cfy.adaptation.filename= ['$TMP_DIR/bbci_classifier_cspp_C3z4_' patch '_' bandstr '_' tag_list{ti} '_pcovmean'];
  
  % FBACK parameter to change
  fb.classes= classes;
  fb.classesMarkers= [int16(clidx1) int16(clidx2)];

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
  
% newblock, Run 2 - Train 'CSP24channels + 6 Patches' on Feedback Run 1
bbci= bbci_default;
bbci.setup= 'cspp_auto';
bbci.setup_opts= [];
bbci.train_file{ti}= strcat(bbci.subdir, '/imag_fbarrow_cspp_C3z4_pcovmean*');
bbci.setup_opts.classes= bbci.classes;
bbci.setup_opts.model= {'RLDAshrink', 'scaling', 1};
% TODO: check the channels 
bbci.setup_opts.clab_csp= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci.setup_opts.band= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.patch= 'auto';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp24_run1');
bbci_bet_prepare
bbci_bet_analyze  
fprintf(['Type ''dbcont'' to save classifier and proceed. Or run again bbci_bet_analyze to\n' ...
  'watch the plots again and select the parameters\n']);
fprintf(['Usual parameters to choose are all in bbci.setup_opts: band, ival, (nPat, nPat_csp to be decreased\n' ...
  'in case of overfitting). If you choose the band, put usually ival and patch equal\n' ...
  'to auto. If you choose ival, put band and patch equal to auto, or leave patch as it\n' ...
  'is, especially if it a not so small patch, since the cross validation to select the patch\n' ...
  'is time consuming. In case of too small ival (1000 ms), put ival = [750 3750] and band\n' ...
  'and patch equal to auto. You might put also class equal to auto, in bbci.classes and\n' ...
  'bbci.setup_opts.classes. You might choose not to visualize the scalp anymore setting\n' ...
  'one or more bbci.setup_opts.visualize_* equal to 0']);

keyboard
close all

% class chosen, CLSTAG can be set, if not yet set
CLSTAG= [upper(bbci.classes{1}(1)) upper(bbci.classes{2}(1))];
clidx1= find(CLSTAG(1)=='LRF');
clidx2= find(CLSTAG(2)=='LRF');


%% Adaptation
bbci.adaptation.fcn= @bbci_adaptation_cspp;
bbci.adaptation.param= {struct('ival', bbci.analyze.ival, 'featbuffer', fv, 'mrk_start', [clidx1 clidx2])};
clear fv
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_csp24_retrain_run1';
bbci.adaptation.mode = 'everything_at_once';

bbci_bet_finish;

% in retrain the cspp change
bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};

% FBACK parameter to change
fb.classes = bbci.classes;
fbint.classesMarkers= [clidx1 clidx2];
fbint.trialsPerClass= 32;
fbint.pauseAfter= 16;

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('setint',fbint);
pyff('save_settings', [pyff_fb_setup '_retrain']);

% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_csp24_retrain');
bbci_apply(bbci);

pause(1);
bbci_acquire_bv('close');
pyff('stop');
pyff('quit'); 

%% newblock - Run 3 - BBCI adaptive Feedback, classifier from above, adaptation by retrain
% Train 'CSP32channels + 6 Patches' on Feedback Run 1 and 2
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_*');
bbci.setup_opts.clab_csp= opt.setup_opts.clab;
% bbci.setup_opts.band= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
bbci.setup_opts.nPat_csp= 3;
bbci.setup_opts.nPat= 3;
% bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp32_run12');
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
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_csp32_retrain_run2';
bbci.adaptation.mode = 'everything_at_once';

bbci_bet_finish;

% in retrain the cspp change
bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('setint',fbint);
pyff('save_settings', [pyff_fb_setup '_retrain']);

%% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_csp32_retrain');
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

%% newblock - Run 4 - BBCI adaptive Feedback, classifier from above, adaptation by retrain
% Train 'CSP32channels + 6 Patches' on Feedback Runs 1, 2 and 3
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_cspp_*');
% bbci.setup_opts.band= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
bbci.setup_opts.nPat_csp= 3;
bbci.setup_opts.nPat= 3;
bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_csp32_run123');
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
bbci.adaptation.filename= '$TMP_DIR/bbci_classifier_cspp_csp32_retrain_run3';
bbci.adaptation.mode = 'everything_at_once';

bbci_bet_finish;

% in retrain the cspp change
bbci.feature.proc= {bbci.cont_proc.proc{1}, bbci.feature.proc{:}};
bbci.cont_proc.proc= {bbci.cont_proc.proc{2}};

pyff('init','FeedbackCursorArrow3');
pause(2)
pyff('set',fb);
pyff('setint',fbint);
pyff('save_settings', [pyff_fb_setup '_retrain']);
%% run
pause(1);
pyff('play','basename', 'imag_fbarrow_cspp_csp32_retrain');
bbci_apply(bbci);

pause(1);
bbci_acquire_bv('close');
pyff('stop');
pyff('quit'); 

%% newblock - Train CSPP for subsequent session
bbci.train_file= {[bbci.subdir, '/imag_fbarrow_cspp_csp*'],};
% bbci.setup_opts.band= 'auto';
% bbci.setup_opts.patch= 'auto';
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.usedPat= 'auto';
bbci.setup_opts.usedPat_csp= 'auto';
bbci.setup_opts.nPat_csp= 3;
bbci.setup_opts.nPat= 3;
bbci.setup_opts = rmfield(bbci.setup_opts, 'patch_centers');
bbci.setup_opts.model= {'RLDAshrink', 'gamma', 0, 'scaling', 1, 'store_means', 1, 'store_invcov',1};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspp_auto_carryover');
fprintf('Decide whether to set band and patch again to auto');
keyboard
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish