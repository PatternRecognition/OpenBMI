%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause


%% Relaxation
fprintf('\n\nRelax recording.\n');
fprintf('Press <RETURN> to start RELAX measurement.\n'); pause;
[seq, wav, opt]= setup_season10_relax;
fprintf('Tell participant to push a button to start relax experiment.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'), close all
stim_artifactMeasurement(seq, wav, opt);

%% Startup pyff
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin" &'])
% system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug" &'])
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', '127.0.0.1', 12345);


%% VEP
fprintf('Press <RETURN> to start VEP measurement.\n'); pause;
setup_VEP;
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','VEP_');
pause(5)
fprintf('Tell participant to push a button to start VEP experiment.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'), close all
send_xmlcmd_udp('interaction-signal', 'command', 'play');
pause(5)
stimutil_waitForMarker('S253');
fprintf('VEP measurement finished!\n')
% bvr_sendcommand('stoprecording');
send_xmlcmd_udp('interaction-signal', 'command', 'stop');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
%% Covert attention practice
fprintf('Press <RETURN> to start practice.\n'); pause;
nPracticeTrials = 20;
setup_covert_attention;
send_xmlcmd_udp('interaction-signal', 'i:nTrials', nPracticeTrials);

send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
pause(5)
fprintf('Tell participant to push a button to practice Visual Attention.\n');
stimutil_waitForMarker({'R  1','R  2'});
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
pause(5)
stimutil_waitForMarker('S253');
fprintf('Practice finished.\n')
% bvr_sendcommand('stoprecording');
send_xmlcmd_udp('interaction-signal', 'command', 'stop');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

%% Covert attention run
fprintf('Press <RETURN> to start covert attention experiment.\n'); pause;

nBlocks = 6;
for ii=1:nBlocks
  setup_covert_attention;
  send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','covert_');
  pause(1)
  fprintf('Press Trigger Button to start run #%d.\n',ii)
  stimutil_waitForMarker({'R  1','R  2'});
  fprintf('Ok, starting...\n'),close all
  send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
  pause(5)
  stimutil_waitForMarker('S253');
%   bvr_sendcommand('stoprecording');
  send_xmlcmd_udp('interaction-signal', 'command', 'stop');
  send_xmlcmd_udp('interaction-signal', 'command', 'quit');
  fprintf('Run #%d finished.\nShort break, press <RET> button to proceed.\n',ii), pause
end

%%
fprintf('Finshed.\n');
