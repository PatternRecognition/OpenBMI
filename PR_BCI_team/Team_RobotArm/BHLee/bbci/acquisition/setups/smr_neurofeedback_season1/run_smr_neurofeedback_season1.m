%% Preparation of the EEG cap
bvr_sendcommand('checkimpedances');
stimutil_waitForInput('msg_next','when finished preparing the cap.');
bvr_sendcommand('viewsignals');
pause(5);

%% Relaxation - eyes open
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Train SMR Extractor on data of relaxation
bbci= [];
bbci.setup= 'smr_extractor';
bbci.func_mrk= 'durchrauschen';
bbci.train_file= strcat(TODAY_DIR, 'resting', VP_CODE);
bbci.save_name= strcat(TODAY_DIR, 'bbci_smr_extractor');
bbci.feedback= '';
bbci_bet_prepare
mrk_orig= mrk;
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.quit_marker= 254;
bbci_bet_finish


%% ** Startup pyff **
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks\SMR_NeuroFeedback" &']);
bvr_sendcommand('viewsignals');
pause(8)
general_port_fields.bvmachine= '127.0.0.1';
general_port_fields.control{3}= 12345;
general_port_fields.feedback_receiver= 'pyff';
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

%% Neurofeedback of SMR amplitude - Run 1
%send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 1 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
%bbci_bet_apply(bbci.save_name, 'modifications',{'bbci.fb_port', 12345});
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
pause(1);

%% Run 2
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 2 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%% Relaxation - eyes open
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Neurofeedback of SMR amplitude - Run 3
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 3 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
pause(1);

%% Run 4
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 4 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%% Relaxation - eyes open
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Neurofeedback of SMR amplitude - Run 5
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 5 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
pause(1);

%% Run 6
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 6 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%% Relaxation - eyes open
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

fprintf('End of this session\n');

%% Run 7 %% to test if it makes a difference sich zu verspannen
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
stimutil_waitForInput('msg_next','to start Run 7 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
