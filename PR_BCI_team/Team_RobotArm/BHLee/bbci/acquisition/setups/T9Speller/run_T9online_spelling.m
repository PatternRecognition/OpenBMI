%% setup everything

settings_bbci= {'bbci.start_marker', 252, ...
                  'bbci.quit_marker', 253, ...
                  'bbci.adaptation.running',0};

bbci_cfy= [TODAY_DIR '/bbci_classifier.mat'];
send_xmlcmd_udp('init', '127.0.0.1', 12345);


 bvr_startrecording('ImpDum'); 
 pause(1) 
 bvr_sendcommand('stoprecording');


 
%% 1. spelling run: standard SubtrialGeneration & sentance "Klaus geht zur Uni"
 fprintf('Start first Speller run. First, we present all Stimuli then start the run by pressing <ENT> \n \n');
 pause
 
 system('cmd /C "E: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\T9Speller & python TrialPresentation.py &');

pause
i = i+1 
bvr_startrecording(['T9Speller_SpellerRun' VP_CODE], 'impedances', 0); 
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'T9Speller','command','sendinit');
pause(2);
send_xmlcmd_udp('interaction-signal', 'i:spellerMode', true );
send_xmlcmd_udp('interaction-signal', 'i:adaptiveSequence', false );
%for testing, simulate sbj!!!
send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false ); 

send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [tmpLogDir 'SpellerRun' num2str(i) '_stdSeq.log'] );

pause(1)

send_xmlcmd_udp('interaction-signal', 'command', 'play');

fprintf('Speller Run started. This it is stopped automatically when user selects EXIT')
%for testing, don't apply online stuff
bbci_bet_apply(bbci_cfy, settings_bbci{:});
%pause

send_xmlcmd_udp('interaction-signal', 'command', 'quit');
bvr_sendcommand('stoprecording');

fprintf('First speller run is done!!! SHORT PAUSE FOR THE SUBJECT TO RELAX! \n \n');
fprintf('Press <ENT> to continue with the 2. run')

pause


%% 2. spelling run: standard SubtrialGeneration & sentance "Franz jagt im Taxi quer durch Berlin"
fprintf('Start second Speller run. First, we present all Stimuli then start the run by pressing <ENT> \n \n'); 
pause
 
 system('cmd /C "d: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\T9Speller & python TrialPresentation.py &');

pause

i = i+1 %for the log file!
bvr_startrecording(['T9Speller_SpellerRun' VP_CODE], 'impedances', 0); 
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'T9Speller','command','sendinit');
pause(1);
send_xmlcmd_udp('interaction-signal', 'i:spellerMode', true );
send_xmlcmd_udp('interaction-signal', 'i:adaptiveSequence', true );
%for testing, simulate sbj!!!
send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false ); 

send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [tmpLogDir 'SpellerRun' num2str(i) '_adaptSeq.log'] );

pause(1)

send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Speller Run started. This it is stopped automatically when user selects EXIT')

%for testing, don't apply online stuff
bbci_bet_apply(bbci_cfy, settings_bbci{:});

%pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
bvr_sendcommand('stoprecording');

pause

