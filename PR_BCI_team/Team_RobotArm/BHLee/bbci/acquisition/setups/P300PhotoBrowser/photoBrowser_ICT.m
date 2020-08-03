% startup_bbcilaptop;
% startup_bbcilaptop06;
% setup_bbci_online;
global general_port_fields TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE acquire_func
acquire_func = @acquire_sigserv;

input('DID YOU REALLY, REALLY DEBLOCK THE PARALLEL PORT?');

addpath([BCI_DIR '\acquisition\stimulation\photobrowser']);

acq_makeDataFolder('multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';

try
    send_xmlcmd_udp();
end


send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
%  system(['cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --port=0x0' dec2hex(IO_ADDR) ' --nogui" &']);
% system(['cmd /C "e: & cd \svn\pyff\src & python FeedbackController.py --port=0x0' dec2hex(IO_ADDR) ' -a e:\svn\bbci\python\pyff\src\Feedbacks" &']);

%% RUN CALIBRATION SESSION
clear fbsettings

% screen position
fbsettings.screen_x = -1280;
fbsettings.screen_y = 0;
fbsettings.screen_width = 1280;
fbsettings.screen_height = 1024;

% protocol
fbsettings.num_blocks = 2;
fbsettings.num_iterations = 10;
fbsettings.num_subtrials_per_iteration = 2;
fbsettings.num_trials = 5;

% timing
fbsettings.cue_presentation_duration = 4000;
fbsettings.inter_block_countdown_duration = 3000;
fbsettings.inter_block_pause_duration = 10000;
fbsettings.inter_stimulus_duration = 100;
fbsettings.inter_trial_duration = 2000;
fbsettings.post_cue_presentation_pause_duration = 1000;
fbsettings.pre_cue_pause_duration = 1000;
fbsettings.result_presentation_duration = 2000;
fbsettings.stimulus_duration = 100;

% general
fbsettings.display_selection_enable = true;
fbsettings.mask_sequence_enable = true;
fbsettings.online_mode_enable = false;
fbsettings.row_col_eval_enable = true;
fbsettings.show_empty_image_frames_enable = false;
fbsettings.startup_sleep_duration = 1;

% stimulus
fbsettings.mask_enable = true;
fbsettings.flash_enable = true;
fbsettings.rotation_enable = true;
fbsettings.scaling_enable = true;

del_db = input('Reset the database (yes/no)? ', 's');
if del_db == 'yes',
  delete('C:\Programme\PhotoBrowserTOBI\p300_database.sqlite');
end

% do the actual calibration 
nrBlocks = 2;
for blockId = 1:nrBlocks,
    fbsettings.target_list = randperm(36)-1;
    fbsettings.target_list = fbsettings.target_list(1:fbsettings.num_trials);

%     bvr_startrecording(['PhotoBrowser_train_' VP_CODE], 'impedances', 0);
            
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300_photobrowser','command','sendinit');
    pause(3);
    fbOpts = fieldnames(fbsettings);
    for optId = 1:length(fbOpts),
        if isnumeric(getfield(fbsettings, fbOpts{optId})),
            send_xmlcmd_udp('interaction-signal', ['i:' fbOpts{optId}], getfield(fbsettings, fbOpts{optId}));
        else
            send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(fbsettings, fbOpts{optId}));
        end
        pause(.005);
    end
    pause(1);
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    if blockId == 1,
        data = storeData(1200, TODAY_DIR, ['PhotoBrowser_train_' VP_CODE]);
    else
        data = storeData(1200, TODAY_DIR, ['PhotoBrowser_train_' VP_CODE '0' num2str(blockId)]);
    end
    stimutil_waitForInput('phrase', 'go');
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
end
clear data


%% analzye data
setup_photobrowser_ICT;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;
close all; clear data;

%% RUN ONLINE SESSION
clear fbsettings

% screen position
fbsettings.screen_x = -1280;
fbsettings.screen_y = 0;
fbsettings.screen_width = 1280;
fbsettings.screen_height = 1024;

% protocol
fbsettings.num_blocks = 1;
fbsettings.num_iterations = 10;
fbsettings.num_subtrials_per_iteration = 2;
fbsettings.num_trials = 100;

% timing
fbsettings.cue_presentation_duration = 4000;
fbsettings.inter_block_countdown_duration = 3000;
fbsettings.inter_block_pause_duration = 10000;
fbsettings.inter_stimulus_duration = 100;
fbsettings.inter_trial_duration = 2000;
fbsettings.post_cue_presentation_pause_duration = 1000;
fbsettings.pre_cue_pause_duration = 1000;
fbsettings.result_presentation_duration = 2000;
fbsettings.stimulus_duration = 100;

% general
fbsettings.display_selection_enable = true;
fbsettings.mask_sequence_enable = true;
fbsettings.online_mode_enable = true;
fbsettings.row_col_eval_enable = true;
fbsettings.show_empty_image_frames_enable = false;
fbsettings.startup_sleep_duration = 1;

% stimulus
fbsettings.mask_enable = true;
fbsettings.flash_enable = true;
fbsettings.rotation_enable = true;
fbsettings.scaling_enable = true;

del_db = input('Reset the database (yes/no)? ', 's');
if del_db == 'yes',
  delete('C:\Programme\PhotoBrowserTOBI\p300_database.sqlite');
end

nrBlocks = 1;
for blockId = 1:nrBlocks,
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300_photobrowser','command','sendinit');
    pause(3);
    fbOpts = fieldnames(fbsettings);
    for optId = 1:length(fbOpts),
        if isnumeric(getfield(fbsettings, fbOpts{optId})),
            send_xmlcmd_udp('interaction-signal', ['i:' fbOpts{optId}], getfield(fbsettings, fbOpts{optId}));
        else
            send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(fbsettings, fbOpts{optId}));
        end
        pause(.005);
    end

    settings_bbci= {'bbci.start_marker', 251, ...
        'bbci.quit_marker', 254, ...
        'bbci.adaptation.running',0};

    bbci_cfy = strcat(TODAY_DIR, 'bbci_classifier.mat');
    pause(1);
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
%     fake_classifier('target', 56, 'val_mrk', [21:60], 'subtrial_mrk', 20, 'mrk_per_subtr', 6)    
%     fake_classifier('target', [1 2 3 30 36]+20, 'val_mrk', [21:60], 'subtrial_mrk', 20, 'mrk_per_subtr', 6)
    bbci_bet_apply(bbci_cfy, settings_bbci{:});
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
    stimutil_waitForInput('phrase', 'go');
end

