% Visual evoced potential after the main experiment

% Brain Vision Recorder must be started with correct workspace
% TODAY_DIR must be set!

RUN_END={'S246' 'S247' 'S255'}
acqFolder = [BCI_DIR 'acquisition/setups/CenterSpellerSequenceEffects/'];
VEP_file = [acqFolder 'VEP_feedback'];
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);

%% VEP checkboard - practice
pyff('init','CheckerboardVEP'); pause(.5)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
pyff('setint','nStim',10);
stimutil_waitForInput('msg_next','to start VEP practice.');
pyff('play');
stimutil_waitForMarker(RUN_END);
fprintf('VEP practice finished.\n')
pyff('quit');

%% VEP checkboard - recording
pyff('init','CheckerboardVEP'); pause(.5);
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
stimutil_waitForInput('msg_next','to start VEP recording.');
pyff('play', 'basename', 'VEP_', 'impendances', 0)
stimutil_waitForMarker(RUN_END);
fprintf('VEP recording finished.\n')
pyff('quit');