
% Online Auditory P300 Speller based on a T9 language system
% @JohannesHoehne 

bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

%testrun
fprintf('Press <RET> to start the test-runs.\n');
%pause()




%TESTRUN
%TODO


%Calibration runs

sbj_counts = 0;
numCalibRuns = 1 %4

fprintf('TESTRUN\n');
% send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
for i=1:numCalibRuns
    bvr_startrecording(['T9SpellerCalibration' VP_CODE], 'impedances', 0); 
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'OnlineAuditoryP300Speller','command','sendinit');
    pause(10);
    send_xmlcmd_udp('interaction-signal', 'i:spellerMode', false );
    pause(1)
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    pause
    %send_xmlcmd_udp('interaction-signal', 'command', 'pause');
    while isnumeric(sbj_counts)
        %catch counts of the trials!!
        sbj_counts = input('enter the counted number (when block completed)', 's');
        if strmatch(sbj_counts,'end')
            break;      
        else
          sbj_counts = str2num(sbj_counts);
          if ~isempty(sbj_counts)
            send_xmlcmd_udp('interaction-signal', 'i:numCounts' , sbj_counts);
          end
        end
    end
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
    bvr_sendcommand('stoprecording');
end
fprintf('Press <RET> to continue with tests measurement.\n');


setup_T9_online;
bbci_bet_prepare;
bbci_bet_analyze;

% only when analyzes satisfiable
bbci_bet_finish;



