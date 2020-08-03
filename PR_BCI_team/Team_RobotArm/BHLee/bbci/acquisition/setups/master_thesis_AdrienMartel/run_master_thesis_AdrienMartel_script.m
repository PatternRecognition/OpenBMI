descW= stimutil_readDescription('am_willkommen');
descA1p= stimutil_readDescription('am_d2practice');
descA1r= stimutil_readDescription('am_d2real');
descA2p= stimutil_readDescription('am_MCpractice');
descA2r= stimutil_readDescription('am_MCreal');

stimutil_showDescription(descW, 'clf',1, 'waitfor', 'key');

RUN_END = {'S253'};

pyff('startup','a',['D:\svn\pyff\src'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(1);

feedback= 'MackworthClock';
feedback2= 'TestD2';
msg= ['to start ' feedback ' '];
msg2= ['to start ' feedback2 ' '];


%% Practice D2
stimutil_showDescription(descA1p, 'clf',1, 'waitfor', 'key');
stimutil_waitForInput('msg_next', [msg2 'practice']);
pyff('init', 'TestD2');
pause(1);
pyff('setint', 'geometry', VP_SCREEN);
pyff('setint','practiceRun', 1);
pyff('play');
stimutil_waitForMarker(RUN_END);
pyff('quit');

%% Real D2
stimutil_showDescription(descA1r, 'clf',1, 'waitfor', 'key');
stimutil_waitForInput('msg_next', [msg ' acquisition']);
pyff('init', 'TestD2');
pause(1);
pyff('setint', 'geometry', VP_SCREEN);
pyff('play', 'basename', 'D2test', 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');
 
%% Practice MC
stimutil_showDescription(descA2p, 'clf',1, 'waitfor', 'key');
stimutil_waitForInput('msg_next', [msg 'practice']);
pyff('init', 'BoringClock');
pause(1);
pyff('setint', 'geometry', VP_SCREEN);
pyff('setint','nClockTicks', 120, 'nJumps', 20, 'practiceRun', 1);
pyff('play');
stimutil_waitForMarker(RUN_END, 'verbose', 1);
pyff('quit');

stimutil_showDescription(descA2r, 'clf',1, 'waitfor', 'key');

for run= 1:4,
%% Acquisition MC
stimutil_waitForInput('msg_next', [msg ' - run ' int2str(run)]);
pyff('init', 'BoringClock');
pause(1);
pyff('setint', 'geometry', VP_SCREEN);
%pyff('setint','nClockTicks', 120, 'nJumps', 20, 'practiceRun', 1);
pyff('setint','nClockTicks', 900, 'nJumps', 45, 'practiceRun', 0);
%ticks 900, jumps 45
pyff('save_settings', feedback);
pyff('play', 'basename', ['calibration_' feedback], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');
end
