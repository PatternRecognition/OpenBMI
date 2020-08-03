

%-newblck
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15] at Lap C3,4), pcovmean adaptation
desc= stimutil_readDescription('season9_imag_fbarrow');
% h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
fprintf('Press <RETURN> when ready to start the next task.\n');
pause
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
fprintf('Record 1 run of feedback then press <EXIT> in the GUI.\n');
system(['matlab -r "' cmd'' BCI_DIR 'acquisition\setups\TOBI_dry_test\setup_MI_dryCapTest;'' matlab_control_gui(''season9/cursor_adapt_pcovmean'', ''classifier'', [TODAY_DIR ''Lap_C34_bp_'' CLSTAG]);" &']);
bbci_bet_apply;

%% - Train CSP-based classifier on Feedback Run 1
bbci= bbci_default;
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_24chans');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1, 'store_invcov',1};
bbci.setup_opts.clab= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci.adaptation.load_tmp_classifier= 0;
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish

%-newblock
%% - BBCI adaptive Feedback, CSP-based classifier, pcovmean adaptation
desc= stimutil_readDescription('season9_imag_fbarrow');
h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
fprintf('Press <RETURN> when ready to start the next task.\n');
pause
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
fprintf('Record 1 run of feedback then press <EXIT> in the GUI.\n');
system(['matlab -r "' cmd 'setup_season9; matlab_control_gui(''season9/cursor_adapt_pcovmean'');" &']);
bbci_bet_apply;

%-newblock
%% - Train CSP-based classifier on Feedback Runs 1, 2
bbci= bbci_default;
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish


%-newblock
%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation RUN 1
desc= stimutil_readDescription('season9_imag_fbarrow');
h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
fprintf('Press <RETURN> when ready to start the next task.\n');
pause
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
fprintf('Record 4 runs of feedback then press <EXIT> in the GUI.\n');
system(['matlab -r "' cmd 'setup_season9; matlab_control_gui(''season9/cursor_adapt_pmean'');" &']);
bbci_bet_apply;


fprintf('Fin\n');
