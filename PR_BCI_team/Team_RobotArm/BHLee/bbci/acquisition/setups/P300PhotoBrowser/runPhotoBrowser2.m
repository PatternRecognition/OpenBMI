startup_bbci;
global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE

acq_getDataFolder('multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';
    
try
   bvr_sendcommand('loadworkspace', ['reducerbox_std_P300.rwksp']);
catch
   error('BrainVision recorder not responding');
end

try
    send_xmlcmd_udp();
end

Highlights = pseudoRandMatrix2(6,5,5,15);

send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --nogui --port=0x2030" &');

clear fbsettings
fbsettings.screenPos = [-1920, 400];
fbsettings.screen_w = 1200;
fbsettings.screen_h = 900;
fbsettings.subtrial_count = 30;
fbsettings.trial_count = 15;
fbsettings.block_count = 2;
fbsettings.inter_stimulus_duration = 1500;
fbsettings.trial_highlight_duration = 2500;
fbsettings.highlight_indexes = squeeze(Highlights(:,:,1));


bvr_startrecording(['impedanceBefore' VP_CODE]);
pause(1);
bvr_sendcommand('stoprecording');
pause(1);

try
    bvr_startrecording(['P300PhotoBrowser_' VP_CODE], 'impedances', 0); 
    pause(2);
catch
    error('The BrainVision recorder is not responding!');
end

send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300PhotoBrowser','command','sendinit');

pause(5);
fbOpts = fieldnames(fbsettings);
for i = 1:length(fbOpts),
    send_xmlcmd_udp('interaction-signal', fbOpts{i}, getfield(fbsettings, fbOpts{i}));
    pause(.005);
end

pause(1);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('When feedback is ready, press <RET>');
pause;
reshape(Highlights,1,450)
send_xmlcmd_udp('interaction-signal', 'command', 'quit');        

pause(1);
bvr_sendcommand('stoprecording');