bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%-newblock
%setup_adaptationstudy_season1_artifacts_demo;
%fprintf('Press <RETURN> when ready to start artifact measurement test.\n');
%pause
%stim_artifactMeasurement(seq, wav, opt, 'test',1);

%-newblock
%setup_adaptationstudy_season1_artifacts;
%fprintf('Press <RETURN> when ready to start artifact measurement.\n');
%pause
%stim_artifactMeasurement(seq, wav, opt);
%fprintf('Press <RETURN> when ready to go to the feedback runs.\n');
%pause

%cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
%fprintf('Record 3 runs of feedback then press <EXIT> in the GUI.\n');
%system(['matlab -r "' cmd 'setup_adaptationstudy_season1; matlab_control_gui(''adaptationstudy_season1/cursor_adapt_pcovmean'', ''classifier'', [EEG_RAW_DIR ''subject_independent_classifiers/Lap_C3z4_bp_'' CLSTAG]);" &']);
%bbci_bet_apply;

%bbci_bet_prepare
%bbci_bet_analyze
%bbci_bet_finish
%close all

cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);

fprintf('Record 3 runs of feedback then press <EXIT> in the GUI.\n');
system(['matlab -r "' cmd 'setup_adaptationstudy_season1; matlab_control_gui(''adaptationstudy_season1/cursor_adapt_lapcsp'');" &'])
bbci_bet_apply

bbci.setup= 'cspauto';
bbci.train_file= strcat(subdir, '/imag_fbarrow_lapcsp*');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto');
bbci.setup_opts.model= {'RLDAshrink', 'scaling',1, 'store_means',1};
bbci.setup_opts.ival= 'auto';
bbci.setup_opts.band= 'auto';
bbci_bet_prepare
bbci_bet_analyze
bbci_bet_finish

fprintf('Record 2 runs of feedback then press <EXIT> in the GUI.\n');
system(['matlab -r "' cmd 'setup_adaptationstudy_season1; matlab_control_gui(''adaptationstudy_season1/cursor_adapt_pmean'');" &'])
bbci_bet_apply
