%% Check impedances 
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause
bvr_sendcommand('viewsignals');
pause(5)

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% ** Startup pyff **
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin" &']);
%system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug" &']);
bvr_sendcommand('viewsignals');
pause(8)
send_xmlcmd_udp('init', '127.0.0.1', 12345);
nPracticeTrials = 20;

%% VEP
setup_VEP;
fprintf('Tell participant to push a button to start VEP experiment.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','vep');
pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
stimutil_waitForMarker('S253');
fprintf('VEP measurement finished.\n')
bvr_sendcommand('stoprecording');
send_xmlcmd_udp('interaction-signal', 'command', 'stop');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Covert attention visual practice
setup_covert_attention;
send_xmlcmd_udp('interaction-signal', 's:mode','vis');
send_xmlcmd_udp('interaction-signal', 'i:nTrials',nPracticeTrials);

fprintf('Tell participant to push a button to practice Visual Attention.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Press <RET> to start visual practice.\n')
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME',''); pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
stimutil_waitForMarker('S253');
fprintf('Practice finished.\n')
%bvr_sendcommand('stoprecording');
%send_xmlcmd_udp('interaction-signal', 'command', 'stop');
%send_xmlcmd_udp('interaction-signal', 'command', 'quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Covert attention visual run 1
setup_covert_attention;
send_xmlcmd_udp('interaction-signal', 's:mode','vis');
fprintf('Tell participant to push a button to start Visual Attention.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','visual');
pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
stimutil_waitForMarker('S253');
fprintf('visual run 1 finished.\n');
bvr_sendcommand('stoprecording');
%send_xmlcmd_udp('interaction-signal', 'command', 'stop');
%send_xmlcmd_udp('interaction-signal', 'command', 'quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Covert attention audio practice
setup_covert_attention;
send_xmlcmd_udp('interaction-signal', 's:mode','aud');
send_xmlcmd_udp('interaction-signal', 'i:nTrials',nPracticeTrials);
fprintf('Tell participant to push a button to practice Audio Attention.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
stimutil_waitForMarker('S253');
fprintf('Practice finished.\n');
% bvr_sendcommand('stoprecording');
%send_xmlcmd_udp('interaction-signal', 'command', 'stop');
%send_xmlcmd_udp('interaction-signal', 'command', 'quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Covert attention audio run 1
setup_covert_attention;
send_xmlcmd_udp('interaction-signal', 's:mode','aud');
fprintf('Tell participant to push a button to start Audio Attention.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','audio');
pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
stimutil_waitForMarker('S253');
fprintf('audio run 1 finished.\n')
bvr_sendcommand('stoprecording');
fprintf('Press <RET> to continue.\n'); pause;
