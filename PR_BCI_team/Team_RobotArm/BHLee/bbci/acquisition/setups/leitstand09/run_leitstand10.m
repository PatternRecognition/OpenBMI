%% Check impedances 
bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

%% Artifacts
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all

%% ** Startup pyff **
system(['cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --nogui -p brainvisionrecorderplugin -a d:\svn\bbci\python\pyff\src\Feedbacks --port=0x' dec2hex(IO_ADDR) '" &']);
% system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -a d:\svn\bbci\python\pyff\src\Feedbacks" &');

% bvr_sendcommand('viewsignals');
pause(10)
general_port_fields.bvmachine = '127.0.0.1';
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

%% Leitstand test
setup_hauptexperiment;
practiceTime = 5*60;
send_xmlcmd_udp('interaction-signal', 'i:common_stop_time',practiceTime);

desc= stimutil_readDescription('leitstand10_hauptexp_test');
h_desc= stimutil_showDescription(desc, 'waitfor',0,'clf',1);
fprintf(['Press <RET> to start Hauptexperiment PRACTICE.\n']);
pause, fprintf('Ok, starting...\n'),close all

send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','leitstand_test_');
pause(5)

send_xmlcmd_udp('interaction-signal', 'command', 'play');
pause(30)
fprintf(['Hauptexperiment practice finished?\nPress <RET> to proceed.\n']);
pause
% bvr_sendcommand('stoprecording');
% send_xmlcmd_udp('interaction-signal', 'command', 'stop');
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');

%% Leitstand experiment (5-mal ausfuehren!)
setup_hauptexperiment;
desc= stimutil_readDescription('leitstand10_hauptexp_run');
h_desc= stimutil_showDescription(desc, 'waitfor',0,'clf',1);
fprintf(['Press <RET> to start Hauptexperiment.\n']);
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','leitstand_');
pause, fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 'command', 'play');
pause(45*60)
fprintf(['Hauptexperiment finished?\nPress <RET> to quit.\n']);
pause,fprintf('Ok, stopping...\n')
% bvr_sendcommand('stoprecording');
send_xmlcmd_udp('interaction-signal', 'command', 'stop');
pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'quit');