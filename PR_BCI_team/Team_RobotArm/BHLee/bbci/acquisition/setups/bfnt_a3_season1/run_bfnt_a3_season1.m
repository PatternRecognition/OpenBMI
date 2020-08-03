bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%-newblock
%% - Artifact measurement: Test recording
fprintf('\n\nArtifact test run.\n');
path([BCI_DIR 'acquisition/setups/season10'], path);
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT TEST measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1);


%-newblock
%% - Artifact measurement: recording
fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
fprintf('Press <RETURN> when ready to go to the RELAX measurement.\n');
pause


%-newblock
setup_bfnt_a3_season1_oddballAuditory_demo;
fprintf('Press <RETURN> when ready to start to the TEST run.\n');
pause
stim_oddballAuditory(N, opt, 'test',1);
 
%-newblock
setup_bfnt_a3_season1_oddballAuditory;
fprintf('Press <RETURN> when ready to start measurement.\n'); 
pause
stim_oddballAuditory(N, opt);


%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> when ready to start RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);


%-newblock
setup_bfnt_a3_season1_oddballAuditory;
fprintf('Press <RETURN> when ready to start measurement.\n'); 
pause
stim_oddballAuditory(N, opt);
