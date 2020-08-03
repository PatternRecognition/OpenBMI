
%%% HandWorkStation Artifacts and Relax Measurement

%%% Artifacts
path([BCI_DIR 'acquisition/setups/season10'], path)
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n'); pause;
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start artifact measurement.\n'); pause;
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all

%%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n'); pause;
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all

