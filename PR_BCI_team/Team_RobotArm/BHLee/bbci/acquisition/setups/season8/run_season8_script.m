%-initblock
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins" &')
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);


%-newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause


%-newblock
%% - Artifact measurement
fprintf('\n\nArtifact test run.\n');
[seq, wav, opt]= setup_season8_artifacts_demo('clstag', CLSTAG);
fprintf('Press <RETURN> when ready to start artifact measurement test.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1);


%-newblock
fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_season8_artifacts('clstag', CLSTAG);
fprintf('Press <RETURN> when ready to start artifact measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Switch off the speakers.\nPress <RETURN> when ready to go to the feedback runs.\n');
pause
close all


%-newblock
%% - Run 1: Cursor arrow with pretrained CFY on 3 Lap with pcovmean adapt
fprintf('\n\nBCI-Run 1 (cursor arrow with pretrained classifier based on 3 fixed Laplacians).\n');
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
fprintf('Record 1 run of feedback, then press <EXIT> in the Matlab-GUI.\n');
system(['matlab -r "' cmd 'setup_season8; matlab_control_gui(''season8/cursor_adapt_pcovmean'', ''classifier'', [TODAY_DIR ''bbci_classifier_sellap_'' VP_CODE ''_setup_001'']);" &']);
bbci_bet_apply;


%-newblock
%% - Test Run for GoalKeeper with keyboard
fprintf('\n\nTest-Run for Keyboard-played Goal Keeper\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [600 300], ...
  'timeUntilIntegration', 0, ...
  'contKeeperMotion', 60, ...
  'timeOfStartAnimation', 200, ...
  'iTimeUntilThreshold', 1, ...
  'showRedBallDuration', 200, ...
  'i:pauseAfter', 40, ...
  'i:trials', 20, ...
  'hitMissDuration', 250);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
pause(3)


%-newblock
%% - Run T1: GoalKeeper with keyboard
fprintf('\n\nKeyboard-Run 1 (Goal Keeper)\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','real_goalkeeper');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [400 300], ...
  'timeUntilIntegration', 0, ...
  'contKeeperMotion', 60, ...
  'timeOfStartAnimation', 200, ...
  'showRedBallDuration', 200, ...
  'iTimeUntilThreshold', 1, ...
  'i:pauseAfter', 40, ...
  'i:trials', 200, ...
  'hitMissDuration', 250);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%-newblock
%% - Train Classifier for Run 2
fprintf('\n\nTrain classifier for BCI-Run 2 (adaptively selected Laplacians).\n');
global bbci_default
bbci= bbci_default;
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.train_file= strcat(subdir, '/imag_fbarrow_sellap_pcovmean*');
bbci.setup_opts.nlaps_per_area= 1;
bbci.setup_opts.select_lap_for_each_band= 1;
bbci.setup_opts.allow_reselecting_laps= 1;
bbci.setup_opts.selband_max= [7 15; 15 35];
bbci.setup_opts.model = {'RLDAshrink', 'scaling',1};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_sellap6_', VP_CODE);
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish


%-newblock
%% - Run 2: Cursor arrow with sbj-specific CFY on 6 Lap with reselect/retrain
fprintf('\nBCI-Run 2 (cursor arrow with classifier based on 6 selected Laplacians).\n');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_sellap6_', VP_CODE);
fprintf('Record 1 run of feedback then press <EXIT> in the Matlab-GUI.\n');
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
system(['matlab -r "' cmd 'setup_season8; matlab_control_gui(''season8/cursor_adapt_sellap'', ''classifier'',''' bbci.save_name '_setup_001'');" &'])
bbci_bet_apply


%-newblock
%% - Run T2: GoalKeeper with keyboard
fprintf('\n\nKeyboard-Run 2 (Goal Keeper)\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','real_goalkeeper');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [400 300], ...
  'timeUntilIntegration', 0, ...
  'contKeeperMotion', 60, ...
  'timeOfStartAnimation', 100, ...
  'showRedBallDuration', 200, ...
  'iTimeUntilThreshold', 1, ...
  'continueAfterMiss', 0, ...
  'i:pauseAfter', 40, ...
  'i:trials', 200, ...
  'hitMissDuration', 250);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%-newblock
%% - Train Classifier for Run 3
fprintf('\nTrain classifier for BCI-Run 3 (CSP).\n');
global bbci_default
bbci= bbci_default;
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.setup= 'cspauto';
bbci.train_file= strcat(subdir, {'/imag_fbarrow_sellap_pcovmean*', '/imag_fbarrow_sellap6_retrain*'});
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_prelim');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci_bet_finish


%-newblock
%% - Run 3: Cursor arrow using CSP with pmean adaptation
fprintf('\nBCI-Run 3 (cursor arrow with classifier based on CSP and pmean adaptation).\n');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_prelim');
fprintf('Record 1 run of feedback then press <EXIT> in the Matlab-GUI.\n');
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
system(['matlab -r "' cmd 'setup_season8; matlab_control_gui(''season8/cursor_adapt_pmean'', ''classifier'',''' bbci.save_name '_setup_001'');" &'])
bbci_bet_apply


%-newblock
%% - Train Classifier for all 3 GoalKeeper runs
fprintf('\nTrain classifier for BCI-Runs 4-6.\n');
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.train_file= strcat(subdir, '/imag_fbarrow*');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci.quit_marker= [253 254];
bbci_bet_finish


%-newblock
%% - Run 4: GoalKeeper with BBCI
fprintf('\n\nBCI-Run 4 (Goal Keeper)\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','imag_fbgoalkeeper');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
general_port_fields.feedback_receiver= 'pyff';
bbci_setup= strcat(TODAY_DIR, 'bbci_classifier_cspauto_setup_001');
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [2500 2000], ...
  'timeUntilIntegration', 700, ...
  'contKeeperMotion', 100, ...
  'timeOfStartAnimation', 500, ...
  'iTimeUntilThreshold', 300, ...
  'hitMissDuration', 1000);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci_setup, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pmean_' today_str];
S= load(bbci_setup);
T= load(tmpfile);
S.cls= T.cls;
save([bbci_setup(1:end-1) '2'], '-STRUCT','S');
pause(3)


%-newblock
%% - Run 5: GoalKeeper with BBCI
fprintf('\n\nBCI-Run 5 (Goal Keeper)\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','imag_fbgoalkeeper');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
general_port_fields.feedback_receiver= 'pyff';
bbci_setup= strcat(TODAY_DIR, 'bbci_classifier_cspauto_setup_002');
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [2000 1500], ...
  'timeUntilIntegration', 600, ...
  'contKeeperMotion', 100, ...
  'timeOfStartAnimation', 500, ...
  'iTimeUntilThreshold', 300, ...
  'hitMissDuration', 1000);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci_setup, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pmean_' today_str];
S= load(bbci_setup);
T= load(tmpfile);
S.cls= T.cls;
save([bbci_setup(1:end-1) '3'], '-STRUCT','S');
pause(3)


%-newblock
%% - Run 6: GoalKeeper with BBCI
fprintf('\n\nBCI-Run 6 (Goal Keeper)\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','imag_fbgoalkeeper');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
general_port_fields.feedback_receiver= 'pyff';
bbci_setup= strcat(TODAY_DIR, 'bbci_classifier_cspauto_setup_003');
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [1750 1250], ...
  'timeUntilIntegration', 500, ...
  'contKeeperMotion', 100, ...
  'timeOfStartAnimation', 500, ...
  'iTimeUntilThreshold', 250, ...
  'hitMissDuration', 1000);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci_setup, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pmean_' today_str];
S= load(bbci_setup);
T= load(tmpfile);
S.cls= T.cls;
save([bbci_setup(1:end-1) '4'], '-STRUCT','S');
pause(3)


%-newblock
%% - Run T3: GoalKeeper with keyboard
fprintf('\n\nKeyboard-Run 3 (Goal Keeper)\nPress <RET> when ready to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','real_goalkeeper');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'GoalKeeper');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', ...
  'durationPerTrial', [400 300], ...
  'timeUntilIntegration', 0, ...
  'contKeeperMotion', 60, ...
  'timeOfStartAnimation', 100, ...
  'showRedBallDuration', 200, ...
  'iTimeUntilThreshold', 1, ...
  'continueAfterMiss', 0, ...
  'i:pauseAfter', 40, ...
  'i:trials', 200, ...
  'hitMissDuration', 250);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%-newblock
%% - Run 7: Cursor arrow using CSP with pmean adaptation
fprintf('\n\nBCI-Run 7 (cursor arrow with classifier based on CSP and pmean adaptation).\n');
general_port_fields.feedback_receiver= 'matlab';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_prelim');
fprintf('Record 1 run of feedback then press <EXIT> in the Matlab-GUI.\n');
cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);
system(['matlab -r "' cmd 'setup_season8; matlab_control_gui(''season8/cursor_adapt_pmean'', ''classifier'',''' bbci.save_name '_setup_001'');" &'])
bbci_bet_apply


%-newblock
fprintf('Do the soft pinball(?).\n');
