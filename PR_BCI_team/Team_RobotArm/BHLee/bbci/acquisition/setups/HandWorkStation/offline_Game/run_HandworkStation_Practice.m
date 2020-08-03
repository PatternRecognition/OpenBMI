

%%% start Pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);


%%% practice
pyff('init','HandWorkStation'); pause(.5)
pyff('set','MODE',1);
pyff('setint','screen_pos',VP_SCREEN);

fprintf('Press <RETURN> to start HandWorkStation Practice\n'); pause;
fprintf('Ok, starting...\n'), close all

pyff('play');
stimutil_waitForMarker(RUN_END);

fprintf('HandWorkStation Practice finished.\n')
pyff('quit'); pause(1);


