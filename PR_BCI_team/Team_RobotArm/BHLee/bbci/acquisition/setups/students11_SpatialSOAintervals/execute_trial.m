function execute_trial(SOA, target_idx, cue_delay, VP_CODE, catchTrial, numIterations)

if catchTrial
    numIterations = 2;
    target_idx = randperm(9);
    target_idx = target_idx(1); %randomized target_index for catch_trials !!
else
    file_name = sprintf('spatialSOAintervals__SOA_%d__%s', SOA, VP_CODE);
    bvr_startrecording(file_name, 'impedances', 0);
    pause(1)
end

cue_delay_submit=  round(1000*cue_delay) - 1000; % internal delay of 1s

if(cue_delay_submit < 0)
    error('cue_delay has to be >1 to be positive')
end

send_xmlcmd_udp('interaction-signal', 'i:ISI' , SOA);
send_xmlcmd_udp('interaction-signal', 'i:keysToSpell', [target_idx]);
send_xmlcmd_udp('interaction-signal', 'i:PRIMING_SPELLING_OFFSET' , cue_delay_submit);
send_xmlcmd_udp('interaction-signal', 'i:N_MARKER_SEQ' , numIterations);
pause(0.5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
send_xmlcmd_udp('interaction-signal', 'i:paused' , 0);
proceedToNextTrial = 0;
while ~proceedToNextTrial
    sbj_counts = input('Enter the counted number (when block completed)', 's');
    sbj_counts = str2num(sbj_counts);
    if ~isempty(sbj_counts)
        if (sbj_counts < 20 && sbj_counts >= 0)
            send_xmlcmd_udp('interaction-signal', 'i:numCounts' , sbj_counts);
            proceedToNextTrial = 1;
        else
            disp('this weird count is not accepted... it might lead to a misleading marker!')
        end
    end
end
pause(2)
disp('Trial beendet')
fprintf('\n')
bvr_sendcommand('stoprecording');
