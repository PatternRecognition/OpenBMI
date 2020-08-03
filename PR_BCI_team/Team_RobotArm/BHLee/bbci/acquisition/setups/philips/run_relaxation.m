%% Relaxation
fprintf('\n\nRelax recording.\n');
fprintf('Press <RETURN> to start RELAX measurement.\n'); pause;
[seq, wav, opt]= setup_season10_relax;
fprintf('Tell participant to push a button to start relax experiment.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'), close all
stim_artifactMeasurement(seq, wav, opt);