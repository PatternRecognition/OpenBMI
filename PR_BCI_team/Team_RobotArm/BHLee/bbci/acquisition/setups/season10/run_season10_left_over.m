%% this would be too much:


%-newblock
%% - Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> when ready to start final RELAX measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);
close all

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
