VP_SCREEN = [0 0 1920 1200];

%% Check impedances 
%bvr_sendcommand('checkimpedances');
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
%system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p FeedbackControllerPluginsBrainAtWork" &']);
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug" &']);
% bvr_sendcommand('viewsignals');
pause(10)
send_xmlcmd_udp('init', '127.0.0.1', 12345);
nPracticeTrials = 20;

%% VEP
% fprintf('Press <RET> to ssetup.\n')
% pause
setup_VEP;
% fprintf('Press <RET> to start VEP measurement.\n')
% pause, fprintf('Ok, starting...\n'),close all
% send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
% send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','VEP_');
% send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','VEP_');
% pause(5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
% pause(30)
% fprintf('VEP measurement finished.\n')
% pause
% bvr_sendcommand('stoprecording');
send_xmlcmd_udp('interaction-signal', 'command', 'stop');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

%% Oddball run

nBlocks = 3; nStims = [100 100 100];
%nStims = [10 15 9];
for nn=1:nBlocks
    setup_oddball;
    send_xmlcmd_udp('interaction-signal', 'i:nStim',nStims(nn));
    send_xmlcmd_udp('interaction-signal', 'i:nStim_per_block ',nStims(nn));

    fprintf(['Press <RET> to start oddball block ' num2str(nn) '.\n']);
    pause, fprintf('Ok, starting...\n'),sleep(3),close all

    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    fprintf(['Oddball block ' num2str(nn) ' finished.\nType "return" and press <RET> to proceed.\n']);
    pause
%     bvr_sendcommand('stoprecording');
end
pause(30)
fprintf(['Oddball measurements completed.\nPress <RET> to proceed.\n']);
pause
