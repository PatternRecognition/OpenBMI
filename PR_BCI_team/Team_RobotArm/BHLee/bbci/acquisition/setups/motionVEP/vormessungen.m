%% Artifacts
% [seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
% fprintf('Press <RETURN> to start TEST artifact measurement.\n');
% pause; fprintf('Ok, starting...\n');
% stim_artifactMeasurement(seq, wav, opt, 'test',1);
% [seq, wav, opt]= setup_season10_artifacts('clstag', '');
% fprintf('Press <RETURN> to start artifact measurement.\n');
% pause; fprintf('Ok, starting...\n');
% stim_artifactMeasurement(seq, wav, opt);

pyff('startup');

%% Standard oddball PRACTICE 
fprintf('Press <RETURN> to start oddball PRACTICE.\n');
pause
setup_oddball
pyff('set','nStim',20)
pyff('setdir','')
pause(.01)
fprintf('Ok, starting...\n'),close all
pyff('play')
pause(5)
stimutil_waitForMarker('stopmarkers','S253');
fprintf('Practice finished? If yes, press <RETURN>\n'),pause
pyff('stop');pause(1);
pyff('quit')

%% Standard oddball measurement 
fprintf('Press <RETURN> to start oddball measurement.\n');
pause
setup_oddball
pyff('setdir','basename','oddball')
pause(.01)
fprintf('Ok, starting...\n'),close all
pyff('play')
pause(5)
stimutil_waitForMarker('stopmarkers','S253');
fprintf('Measurement finished? If yes, press <RETURN>\n'),pause
pyff('stop');pause(1);
pyff('quit')
bvr_sendcommand('stoprecording');


%%
fprintf('Vormessungen finished.\n');