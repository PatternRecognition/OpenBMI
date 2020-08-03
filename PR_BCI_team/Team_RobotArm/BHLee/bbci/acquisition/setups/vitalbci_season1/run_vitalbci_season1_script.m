bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

fprintf('Ask the subject to fill the first 2 tests. Press <RETURN> when finished.\n');
pause

%-newblock
setup_vitalbci_season1_artifacts_demo;
fprintf('Press <RETURN> when ready to start artifact measurement test.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1);

%-newblock
setup_vitalbci_season1_artifacts;
fprintf('Press <RETURN> when ready to start artifact measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the next run (movement observation).\n');
pause

%-newblock
setup_vitalbci_season1_observation;
fprintf('Press <RETURN> when ready to start ''observed movements'' measurement.\n');
pause
stim_videoCues(stim, opt);
fprintf('Press <RETURN> when ready to go to the next run (real movements).\n');
pause

%-newblock
setup_vitalbci_season1_real_arrow;
fprintf('Press <RETURN> when ready to start ''real movements'' test.\n');
pause
stim_visualCues(stim, opt, 'test',1);
fprintf('Press <RETURN> when ready to start the recording.\n');
pause

%-newblock
setup_vitalbci_season1_real_arrow;
fprintf('Press <RETURN> when ready to start ''real movements'' measurement.\n');
pause
stim_visualCues(stim, opt);
fprintf('Press <RETURN> when ready to go to the next run (imagined movements).\n');
pause

%-newblock
setup_vitalbci_season1_imag_arrow;
fprintf('Press <RETURN> when ready to start ''imagined movements'' test.\n');
pause
stim_visualCues(stim, opt, 'test',1);
fprintf('Press <RETURN> when ready to start the recording.\n');
pause

%-newblock
setup_vitalbci_season1_imag_arrow;
fprintf('Press <RETURN> when ready to start ''imagined movement'' measurement.\n');
pause
stim_visualCues(stim, opt);
fprintf('Ask the subject to fill the next test; Press <RETURN> when ready to go to the next run (d2-test).\n');
pause

%-newblock
setup_vitalbci_season1_d2test_demo;
fprintf('Press <RETURN> when ready to start slow d2-test test.\n');
pause
stim_d2test(15, opt, 'test',1);

%-newblock
setup_vitalbci_season1_d2test;
fprintf('Press <RETURN> when ready to start fast d2-test test.\n');
pause
stim_d2test(10, opt, 'test',1);

%-newblock
setup_vitalbci_season1_d2test;
fprintf('Press <RETURN> when ready to go to the real d2-test recording.\n');
pause
stim_d2test(N, opt);
fprintf('Press <RETURN> when ready to go to the next run (imagined movements).\n');
pause

%-newblock
setup_vitalbci_season1_imag_arrow;
fprintf('Press <RETURN> when ready to start ''imagined movements'' measurement.\n');
pause
stim_visualCues(stim, opt);

fprintf('Ask the subject to fill the next test.\nPress <RETURN> when ready to go to the next run (d2-test).\n');
pause

%-newblock
setup_vitalbci_season1_d2test;
fprintf('Press <RETURN> when ready to start next d2-test recording.\n');
pause
stim_d2test(N, opt);
fprintf('Press <RETURN> when ready to go to the next run (imagined movements).\n');
pause

%-newblock
setup_vitalbci_season1_imag_arrow;
fprintf('Press <RETURN> when ready to start ''imagined movements'' measurement.\n');
pause
stim_visualCues(stim, opt);

fprintf('Ask the subject to fill the next test; Press <RETURN> when finished to start the training.\n');
pause

fprintf('You still have to do:\n');
fprintf('bbci_bet_prepare\n');
fprintf('bbci_bet_analyze\n');
fprintf('bbci_bet_finish\n');
fprintf('bbci_bet_apply\n');

return

% setup_vitalbci_season1_observation;
% fprintf('Press <RETURN> when ready to start ''observed movements'' measurement.\n');
% pause
% stim_videoCues(stim, opt);
