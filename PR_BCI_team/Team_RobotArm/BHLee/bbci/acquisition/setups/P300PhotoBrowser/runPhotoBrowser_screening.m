% startup_bbcilaptop;
% startup_bbcilaptop06;
% setup_bbci_online;
global general_port_fields TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE

addpath([BCI_DIR '\acquisition\stimulation\photobrowser']);

acq_makeDataFolder('multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';
    
% try
%    bvr_sendcommand('loadworkspace', ['reducerbox_64std_visual.rwksp']);
% catch
%    error('BrainVision recorder not responding');
% end

try
    send_xmlcmd_udp();
end

modes = {'RowSimple','RowComplex','OptimSimple','OptimComplex'};

optimal = load([BCI_DIR '\acquisition\stimulation\photobrowser\Seq_Screensize_6x6_GroupSize_6_Frames_15.mat']);
csvFilename = [BCI_DIR '\python\pyff\src\Feedbacks\P300PhotoBrowser\highlights.csv'];

send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
%  system(['cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --port=0x0' dec2hex(IO_ADDR) ' --nogui" &']);
% system(['cmd /C "e: & cd \svn\pyff\src & python FeedbackController.py --port=0x0' dec2hex(IO_ADDR) ' -a e:\svn\bbci\python\pyff\src\Feedbacks" &']);
% to past to cmd:
%   e:
%   cd e:\svn\pyff\src
%   python FeedbackController.py --port=0x05C00 -a e:\svn\bbci\python\pyff\src\Feedbacks

%% RUN CALIBRATION SESSION
clear fbsettings
fbsettings.screenPos = [-1920, 0];
fbsettings.screen_w = 1920;
fbsettings.screen_h = 1200;
% fbsettings.screenPos = [100, 100]; % for testing
% fbsettings.screen_w = 800; % for testing
% fbsettings.screen_h = 600; % for testing
fbsettings.subtrial_count = 60;
fbsettings.trial_count = 6; %6
fbsettings.block_count = 3; %3
fbsettings.highlight_count = 6;
fbsettings.inter_stimulus_duration = 120;
fbsettings.stimulation_duration = 100;
fbsettings.trial_pause_duration = 500;
fbsettings.inter_block_duration = 15000;
fbsettings.trial_highlight_duration = 2500;
fbsettings.trial_pause_duration = 5000;
fbsettings.highlight_all_selected = false;
fbsettings.online_mode = false;
fbsettings.copy_task = 1;
fbsettings.startup_sleep = 1;
fbsettings.row = 6;
fbsettings.col = 6;
fbsettings.image_display = false;
nrFrames = fbsettings.subtrial_count / fbsettings.highlight_count;
nrIdx = fbsettings.subtrial_count*fbsettings.highlight_count;

nrBlocks = 2;
modeId = randperm(length(modes));
% modeId = repmat(modeId, 1, nrBlocks);
save([TODAY_DIR 'offlineOrder.mat'], 'modeId');

targetList = [0:fbsettings.row*fbsettings.col-1];
% targetList = repmat(targetList, 1, mod(nrBlocks*fbsettings.trial_count*fbsettings.block_count, length(targetList)-1));
targetList = targetList(randperm(length(targetList)));
targetList = reshape(targetList, nrBlocks, []);


for blockId = 1:nrBlocks,
    disp(sprintf('Block: %i',blockId));
    try
        bvr_startrecording(['impedanceBefore' VP_CODE]);
        pause(1);
        bvr_sendcommand('stoprecording');
        pause(1);
    catch
        error('The BrainVision recorder is not responding!');
    end

    blockTargets = targetList(blockId, :);

    for i = 1:length(modeId),
        disp(sprintf('Run: %i',i));
        currMode = modeId(i);
        if ~isempty(strmatch('Row', modes{currMode})),
            fbsettings.rowColEval = true;
            flashes = zeros(100, nrIdx);
            for seqi = 1:100;
                flashes(seqi,:) = createRowColumnSequences(5, 6, 6, 2)-1;
            end
        else
            fbsettings.rowColEval = false;
            Sequences = optimal.Sequences;
            select_seq = randperm(1000);
            select_seq = select_seq(1:100);
            flashes = zeros(length(select_seq), numel(Sequences{1}.seq));
            for trNr=1:length(select_seq),
                flashes(trNr,:) = reshape(permute(Sequences{select_seq(trNr)}.seq, [2 1 3]), 1, [])-1;
            end
        end
        flashes = flashes(:,1:nrIdx);
        csvwrite(csvFilename, flashes);

        if ~isempty(regexp(modes{currMode}, 'Simple')),
            fbsettings.rotation = false;
            fbsettings.mask = false;
            fbsettings.enlarge = false;
        else
            fbsettings.rotation = true;
            fbsettings.mask = true;
            fbsettings.enlarge = true;
        end
        fbsettings.target_list = blockTargets(randperm(length(blockTargets)));
        disp(sprintf('Targets used: %s',num2str(fbsettings.target_list)));
        disp(sprintf('Mode: %s',modes{currMode}));

        bvr_startrecording(['PhotoBrowser_train_' modes{currMode} VP_CODE], 'impedances', 0);

        send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300PhotoBrowser','command','sendinit');
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
        stimutil_waitForMarker('S254');
        bvr_sendcommand('stoprecording');
        stimutil_waitForInput('phrase', 'go');
        send_xmlcmd_udp('interaction-signal', 'command', 'quit');
    end
end


%% analzye data
setup_photobrowser_online;

bbci.train_file = strcat(subdir, '/PhotoBrowser_train_OptimSimple',VP_CODE);
bbci.save_name = strcat(TODAY_DIR, 'OptimSimpleClassifier');

bbci.train_file = strcat(subdir, '/PhotoBrowser_train_OptimComplex',VP_CODE);
bbci.save_name = strcat(TODAY_DIR, 'OptimComplexClassifier');

bbci.train_file = strcat(subdir, '/PhotoBrowser_train_RowSimple',VP_CODE);
bbci.save_name = strcat(TODAY_DIR, 'RowSimpleClassifier');

bbci.train_file = strcat(subdir, '/PhotoBrowser_train_RowComplex',VP_CODE);
bbci.save_name = strcat(TODAY_DIR, 'RowComplexClassifier');

% bbci.impedance_threshold = Inf;
% bbci.withclassification = 0; bbci.withgraphics = 0;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;
close all;

%% RUN ONLINE SESSION
clear fbsettings
fbsettings.screenPos = [-1920, 0];
fbsettings.screen_w = 1920;
fbsettings.screen_h = 1200;
% fbsettings.screenPos = [100, 100]; % for testing
% fbsettings.screen_w = 800; % for testing
% fbsettings.screen_h = 600;
fbsettings.subtrial_count = 60;
fbsettings.trial_count = 6;
fbsettings.block_count = 3;
fbsettings.highlight_count = 6;
fbsettings.inter_stimulus_duration = 120;
fbsettings.stimulation_duration = 100;
fbsettings.trial_highlight_duration = 2500;
fbsettings.trial_pause_duration = 500;
fbsettings.online_mode = true;
fbsettings.highlight_all_selected = true;
fbsettings.copy_task = 1;
fbsettings.startup_sleep = 5;
fbsettings.image_display = true;
fbsettings.image_display_time = 1000;
fbsettings.row = 6;
fbsettings.col = 6;
nrFrames = fbsettings.subtrial_count / fbsettings.highlight_count;
nrIdx = fbsettings.subtrial_count*fbsettings.highlight_count;


nrBlocks = 1;

modeId = randperm(length(modes));
% modeId = repmat(modeId, 1, nrBlocks);
save([TODAY_DIR 'offlineOrder.mat'], 'modeId');

targetList = [0:fbsettings.row*fbsettings.col-1];
% targetList = repmat(targetList, 1, mod(nrBlocks*fbsettings.trial_count*fbsettings.block_count, length(targetList)-1));
targetList = targetList(randperm(length(targetList)));
targetList = targetList(1:18);

for blockId = 1:nrBlocks,
    disp(sprintf('Run: %i',i));
    try
        bvr_startrecording(['impedanceBefore' VP_CODE]);
        pause(1);
        bvr_sendcommand('stoprecording');
        pause(1);
    catch
        error('The BrainVision recorder is not responding!');
    end

    blockTargets = targetList(blockId, :);

    for i = 1:length(modeId),
        disp(sprintf('\nRun: %i',i));
        currMode = modeId(i);
        if ~isempty(strmatch('Row', modes{currMode})),
            fbsettings.rowColEval = true;
            flashes = zeros(100, nrIdx);
            for seqi = 1:100;
                flashes(seqi,:) = createRowColumnSequences(5, 6, 6, 2)-1;
            end
        else
            fbsettings.rowColEval = false;
            Sequences = optimal.Sequences;
            select_seq = randperm(1000);
            select_seq = select_seq(1:100);
            flashes = zeros(length(select_seq), numel(Sequences{1}.seq));
            for trNr=1:length(select_seq),
                flashes(trNr,:) = reshape(permute(Sequences{select_seq(trNr)}.seq, [2 1 3]), 1, [])-1;
            end
        end
        flashes = flashes(:,1:nrIdx);
        csvwrite(csvFilename, flashes);

        if ~isempty(regexp(modes{currMode}, 'Simple')),
            fbsettings.rotation = false;
            fbsettings.mask = false;
            fbsettings.enlarge = false;
        else
            fbsettings.rotation = true;
            fbsettings.mask = true;
            fbsettings.enlarge = true;
        end
        fbsettings.target_list = blockTargets(randperm(length(blockTargets)));
        disp(sprintf('\nTargets used: %s',num2str(fbsettings.target_list)));
        disp(sprintf('\nMode: %s',modes{currMode}));
        disp(sprintf('\nClassifier used: %s',[modes{currMode} 'Classifier.mat']));

        bvr_startrecording(['PhotoBrowser_online_' modes{currMode} VP_CODE], 'impedances', 0);

        send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300PhotoBrowser','command','sendinit');
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

%         bbci_cfy= [TODAY_DIR '\bbci_classifier.mat'];
        bbci_cfy = strcat(TODAY_DIR, modes{currMode}, 'Classifier.mat');
        pause(1);
        send_xmlcmd_udp('interaction-signal', 'command', 'play');
        bbci_bet_apply(bbci_cfy, settings_bbci{:})
%         fake_classifier(struct(), 'target', 26, ...
%             'response_delay', 800, ...
%             'val_mrk', [21:56 121:156], ...
%             'subtrial_mrk', 20, ...
%             'mrk_per_subtr', 6);
%         pause(1);
        bvr_sendcommand('stoprecording');
        send_xmlcmd_udp('interaction-signal', 'command', 'quit');
        stimutil_waitForInput('phrase', 'go');
    end
end

