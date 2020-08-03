startup_bbcilaptop;
startup_bbcilaptop06;
setup_bbci_online;
global general_port_fields TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE

acq_getDataFolder('multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';
    
% try
%    bvr_sendcommand('loadworkspace', ['reducerbox_std_P300.rwksp']);
% catch
%    error('BrainVision recorder not responding');
% end

try
    send_xmlcmd_udp();
end

load('C:\svn\bbci\acquisition\stimulation\photobrowser\Seq_Screensize_6x5_GroupSize_5_Frames_15.mat');
csvFilename = 'C:\svn\pyff\src\Feedbacks\P300PhotoBrowser\highlights.csv';

send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
% system(['cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --port=0x0' dec2hex(IO_ADDR) ' --nogui" &']);
system(['cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --port=0x0' dec2hex(IO_ADDR) ' --nogui" &']);

%% RUN CALIBRATION SESSION
clear fbsettings
% fbsettings.screenPos = [-1510, 150];
% fbsettings.screen_w = 1100;
% fbsettings.screen_h = 900;
% fbsettings.subtrial_count = 90;
% fbsettings.trial_count = 5;
% fbsettings.block_count = 2;
% fbsettings.highlight_count = 5;
% fbsettings.inter_stimulus_duration = 200;
% fbsettings.trial_highlight_duration = 2500;
% fbsettings.trial_pause_duration = 2000;
% fbsettings.online_mode = 0;
% fbsettings.copy_task = 1;
% fbsettings.startup_sleep = 1;
% blocks = 4;
fbsettings.screenPos = [-1510, 150];
fbsettings.screen_w = 1100;
fbsettings.screen_h = 900;
fbsettings.subtrial_count = 90;
fbsettings.trial_count = 5;
fbsettings.block_count = 2;
fbsettings.highlight_count = 5;
fbsettings.inter_stimulus_duration = 200;
fbsettings.trial_highlight_duration = 3500;
fbsettings.trial_pause_duration = 2000;
fbsettings.online_mode = 0;
fbsettings.copy_task = 1;
fbsettings.startup_sleep = 1;
blocks = 4;

try
  bvr_startrecording(['impedanceBefore' VP_CODE]);
  pause(1);
  bvr_sendcommand('stoprecording');
  pause(1);
catch
  error('The BrainVision recorder is not responding!');
end

for i = 1:blocks,
  select_seq = randperm(1000);
  select_seq = select_seq(1:100);
  flashes = zeros(100, numel(Sequences{1}.seq));
  for trNr=1:length(select_seq),
     flashes(trNr,:) = reshape(permute(Sequences{select_seq(i)}.seq, [2 1 3]), 1, [])-1;
  end
  csvwrite(csvFilename, flashes);

  bvr_startrecording(['P300PhotoBrowser_train_' VP_CODE], 'impedances', 0); 
  
  send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300PhotoBrowser','command','sendinit');
  pause(3);
  fbOpts = fieldnames(fbsettings);
  for optId = 1:length(fbOpts),
    send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(fbsettings, fbOpts{optId}));
    pause(.005);
  end
  pause(1);
  send_xmlcmd_udp('interaction-signal', 'command', 'play');
  stimutil_waitForMarker('S254');
  bvr_sendcommand('stoprecording');
  stimutil_waitForInput('phrase', 'go');
  send_xmlcmd_udp('interaction-signal', 'command', 'quit');  
end


%% analzye data
setup_photobrowser_online;
% bbci.impedance_threshold = Inf;
% bbci.withclassification = 0; bbci.withgraphics = 0;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;
close all;

%% RUN ONLINE SESSION
clear fbsettings
% fbsettings.screenPos = [-1510, 150];
% fbsettings.screen_w = 1100;
% fbsettings.screen_h = 900;
% fbsettings.subtrial_count = 90;
% fbsettings.trial_count = 5;
% fbsettings.block_count = 2;
% fbsettings.highlight_count = 5;
% fbsettings.inter_stimulus_duration = 200;
% fbsettings.trial_highlight_duration = 3500;
% fbsettings.trial_pause_duration = 1000;
% fbsettings.online_mode = 1;
% fbsettings.copy_task = 0;
% fbsettings.highlight_all_selected = 1;
% fbsettings.startup_sleep = 5;
% blocks = 1;
fbsettings.screenPos = [-1510, 150];
fbsettings.screen_w = 1100;
fbsettings.screen_h = 900;
fbsettings.subtrial_count = 90;
fbsettings.trial_count = 50;
fbsettings.block_count = 1;
fbsettings.highlight_count = 5;
fbsettings.inter_stimulus_duration = 200;
fbsettings.trial_highlight_duration = 1500; %3500
fbsettings.trial_pause_duration = 500; %1000
fbsettings.online_mode = 1;
fbsettings.copy_task = 0;
fbsettings.highlight_all_selected = 1;
fbsettings.startup_sleep = 5;
blocks = 1;

% try
%   bvr_startrecording(['impedanceBefore' VP_CODE]);
%   pause(1);
%   bvr_sendcommand('stoprecording');
%   pause(1);
% catch
%   error('The BrainVision recorder is not responding!');
% end

for i = 1:blocks
  select_seq = randperm(1000);
  select_seq = select_seq(1:100);
  flashes = zeros(100, numel(Sequences{1}.seq));
  for trNr=1:length(select_seq),
     flashes(trNr,:) = reshape(permute(Sequences{select_seq(i)}.seq, [2 1 3]), 1, [])-1;
  end
  flashes = flashes(:,1:fbsettings.subtrial_count*fbsettings.highlight_count);
  csvwrite(csvFilename, flashes);

  bvr_startrecording(['P300PhotoBrowser_online_' VP_CODE], 'impedances', 0);   
  
  send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300PhotoBrowser','command','sendinit');
  pause(3);
  fbOpts = fieldnames(fbsettings);
  for optId = 1:length(fbOpts),
    send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(fbsettings, fbOpts{optId}));
    pause(.005);
  end
% 
%   settings_bbci= {'bbci.start_marker', 251, ...
%               'bbci.quit_marker', 254, ...
%               'bbci.adaptation.running',0};
  settings_bbci= {'bbci.adaptation.running',0};
            
  bbci_cfy= [TODAY_DIR '\bbci_classifier.mat'];
  pause(1);
  send_xmlcmd_udp('interaction-signal', 'command', 'play');
  bbci_bet_apply(bbci_cfy, settings_bbci{:})
  pause(1);
  bvr_sendcommand('stoprecording');
  send_xmlcmd_udp('interaction-signal', 'command', 'quit');     
  stimutil_waitForInput('phrase', 'go');
end


