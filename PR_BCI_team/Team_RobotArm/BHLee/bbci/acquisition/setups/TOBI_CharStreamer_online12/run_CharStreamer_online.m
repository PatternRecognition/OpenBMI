% matlab script for Online CharStreamer feedback
pause on;
% TODO brainrecorder stuff

bvr_sendcommand('viewsignals');

send_xmlcmd_udp('init', '127.0.0.1', 12345)
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CharStreamer','command','sendinit')


% IMPORTANT !wait till feedback is completely loaded
stimutil_waitForMarker('S 94')  % !!! wait till feedback is loaded
fprintf('   Initialization complete! \n');
pause(1)
            
iterations = 15;
% configuration
send_xmlcmd_udp('interaction-signal', 'b:calibration_mode' , 'False');
send_xmlcmd_udp('interaction-signal', 'b:online_mode' , 'True');
send_xmlcmd_udp('interaction-signal', 'b:online_simulation' , 'False');
send_xmlcmd_udp('interaction-signal', 'i:iterations' , iterations);
send_xmlcmd_udp('interaction-signal', 'i:pre_iterations' , 0); % fake iterations before

send_xmlcmd_udp('interaction-signal', 'b:early_stopping' , 'True');
send_xmlcmd_udp('interaction-signal', 'i:min_iterations', 6); % for early stopping
send_xmlcmd_udp('interaction-signal', 'f:p_criterion', 0.01);

%% setting logfile
send_xmlcmd_udp('interaction-signal', 's:file_log' , [TODAY_DIR    'feedbackLogger.log']);


%% simulating
% send_xmlcmd_udp('interaction-signal', 'b:online_simulation' , 'True');
% 
% % start - !!! feedback only stops in simulation mode, otherwise it loops
% % until stop is hit
% for t = {'z', 'h', 'x', 'les', 'del', 'i', 'leer', 'h', 'o'}
%     t
%     send_xmlcmd_udp('interaction-signal', 's:target', char(t)); % works only for simulation and calibration
%     pause(1)
%     send_xmlcmd_udp('interaction-signal', 'command', 'play'); % so this isn't needed in normal online mode
%     pause
% end


%% setup BCI

%% Train the classifier

bbci.calibrate.file = strcat(['CharStreamer_condi' condi_description VP_CODE '*'])
bbci.calibrate.save.file= strcat(['bbci_condi_' condi_description 'classifier_CharStreamer_', VP_CODE]);

 bbci.calibrate.settings.cfy_ival = 'auto'
 %bbci.calibrate.settings.cfy_ival=[ 460 570; ...
% 820 940]   


[bbci, data]= bbci_calibrate(bbci);
bbci= copy_subfields(bbci, bbci_default);
bbci.quit_condition.marker = 254;
bbci_save(bbci, data);


%%
bbci.quit_condition.marker = 254;
fname = ['CharStreamer_onlineRun_' condi_description];
% bvr_startrecoserding(fname, 'impedances', 0);
input('press <ENTER> to start the online phase: ')
pause(2)
send_xmlcmd_udp('interaction-signal', 'command', 'play')
bbci_apply(bbci);
% bvr_sendcommand('stoprecording')




%% stop
send_xmlcmd_udp('interaction-signal', 'command', 'stop');

% quit
send_xmlcmd_udp('interaction-signal', 'command', 'quit');