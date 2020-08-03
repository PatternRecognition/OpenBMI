%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  RUNFILE FOR MULTI-TIMESCALE EXPERIMENT  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization of experiment
if ~exist('INITIALIZED', 'var') || ~INITIALIZED,
    startup_bbci;
    setup_bbci_online;
    global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE
    tmpVP = input('Does the subject have a VP-CODE? If so, enter it here: ', 's');
    if ~isempty(tmpVP),
        VP_CODE = tmpVP;
    end    
    acq_getDataFolder('multiple_folders',1);

    REMOTE_RAW_DIR= TODAY_DIR;
    
    set_general_port_fields('localhost');
    general_port_fields.feedback_receiver = 'pyff';

    load([TODAY_DIR 'tmp_classifier.mat'], 'bbci');
    
    try
       bvr_sendcommand('loadworkspace', ['reducerbox_64mcc_noEOG.rwksp']);
    catch
       error('BrainVision recorder not responding');
    end
    
    try
        send_xmlcmd_udp();
    end
    send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
    system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --nogui --port=0x2030 -a d:\svn\bbci\python\pyff\src\Feedbacks" &');
%     system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --port=0x2030 -a d:\svn\bbci\python\pyff\src\Temp" &');
    
%     clc;
    fprintf('\nInitializing the experiment.\n\n');
    colors = {[  00.0, 50, 50, 100],[ 60.0, 50, 50, 100],[120.0, 50, 50, 100],[180.0, 50, 50, 100],[240.0, 50, 50, 100],[300.0, 50, 50, 100]};
            
    train.setting = 'Train';
    train.conditions = {'Multi', 'Single'};
    train.files = {'imag_fb_adaptation', 'imag_fb_adaptation'};
    train.runs = 1; % nr of runs per condition
    train.trials = 20; % nr of trials per run
    train.sequence = repmat([2 1], [1, train.runs]);
    train.colSeq = zeros(length(train.sequence), size(colors, 2));
    for i = 1:length(train.sequence),
        train.colSeq(i,:) = randperm(size(colors, 2));
    end
    clear baseSequence;

    test.setting = 'Test';
    test.conditions = {'Multi','Single'};
    test.files = {'imag_fbmulticlassifier', 'imag_fbsingleclassifier'};
    test.runs = 5; % nr of runs per condition
    test.trials = 40; % nr of trials per run
    test.sequence = repmat(randperm(length(test.conditions)), [1, test.runs]);
    test.colSeq = zeros(length(test.sequence), size(colors, 2));
    for i = 1:length(test.sequence),
        test.colSeq(i,:) = randperm(size(colors, 2));
    end    
    clear baseSequence;
    
    fbopt.y_spread = 4;
    fbopt.x_spread = 8;
    fbopt.trial_duration = 5;
    fbopt.show_cross_duration = 2;
    fbopt.show_arrow_duration = .5;
    fbopt.blank_screen_min_duration = .8;
    fbopt.blank_screen_max_duration = 1.5;
    fbopt.bubble_mode = false;
    fbopt.screen_w = 1920;
    fbopt.screen_h = 1200;
    fbopt.score_display_duration = 20;
    fbopt.score_display_period = 20;
    fbopt.weight_factor = bbci.weightVector; 
    fbopt.bias = [0 0 0 0 0];
    fbopt.gain = 0.1;
   
%     reported_best = zeros(1,length(test.runs/2));
    currentRound = 1;
    currentSetting = 'Prepare';
    roundSucces = false;
    
    BET_APPLY_RUNNING = false;
    INITIALIZED = true;
end

%% Preparation of experiment
if strcmp(currentSetting, 'Prepare'),
    fprintf('Please ensure that Pyff is running and press <RET>.\n');
    pause;
    
    copyfile([TODAY_DIR '\tmp_classifier.mat'],[TODAY_DIR '/adapted_cls.mat'])
    currentSetting = 'Bias';
end

%% Run the 'calibration' rounds
if strcmp(currentSetting, 'Bias'),
    if ~roundSucces,
        trials_to_do = [currentRound:length(train.sequence)];
    else
        trials_to_do = [currentRound+1:length(train.sequence)];
    end
    
    for currentRound = trials_to_do,
       roundSucces = false;
       settings_bbci= {'bbci.start_marker', 252, ...
                    'bbci.quit_marker', 254, ...
                    'bbci.adaptation.policy', 'pcovmean_test', ...
                    'bbci.adaptation.offset', 750, ...
                    'bbci.adaptation.tmpfile', [TODAY_DIR '/adapted_clsTmp.mat'], ...
                    'bbci.adaptation.mrk_start', {1,2}, ...  
                    'bbci.adaptation.mrk_end', [100:163], ...
                    'bbci.adaptation.UC_pcov', 0.03, ...
                    'bbci.adaptation.UC_mean', 0.075, ...
                    'bbci.adaptation.load_tmp_classifier', currentRound>1,...
                    'bbci.adaptation.running',1,...
                    'bbci.filt',[]};

        bbci_cfy= [TODAY_DIR '/adapted_cls.mat'];
 
        fileName = train.files{train.sequence(currentRound)};
        bvr_startrecording(['impedanceBeforeBias' VP_CODE]);
        pause(1);
        bvr_sendcommand('stoprecording');
        pause(1);        
        try
            bvr_startrecording([fileName VP_CODE], 'impedances', 0); 
            pause(2);
        catch
            error('The BrainVision recorder is not responding!');
        end
        fprintf('Training round %i: %s\n', currentRound, train.conditions{train.sequence(currentRound)});
        %%%%%%%%%%%%%%%%%%%% Start the actual experiments
        send_xmlcmd_udp('interaction-signal', 's:_feedback', 'MultiScaleFeedback','command','sendinit');
        pause(5);
        fbOpts = fieldnames(fbopt);
        for i = 1:length(fbOpts),
            send_xmlcmd_udp('interaction-signal', fbOpts{i}, getfield(fbopt, fbOpts{i}));
            pause(.01);
        end
        if strcmp(train.conditions{train.sequence(currentRound)}, 'Single'),
            send_xmlcmd_udp('interaction-signal', 'i:classifier_limit', 1);
        else
            send_xmlcmd_udp('interaction-signal', 'i:classifier_limit', 0);
        end
        send_xmlcmd_udp('interaction-signal', 'i:number_of_trials', 10); 
        send_xmlcmd_udp('interaction-signal', 'i:number_of_trials', train.trials); 
        send_xmlcmd_udp('interaction-signal', 'colours', [colors(train.colSeq(currentRound,:))]); %% How to send this properly
        send_xmlcmd_udp('interaction-signal', 'command', 'play');
        bbci_bet_apply(bbci_cfy, settings_bbci{:})
        send_xmlcmd_udp('interaction-signal', 'command', 'quit'); 
        %%%%%%%%%%%%%%%%%%%% End the actual experiments
        pause(1);
        bvr_sendcommand('stoprecording');
        save([TODAY_DIR 'FeedbackVars' test.conditions{test.sequence(currentRound)} int2str(currentRound) '.mat'], 'bbci', 'fbopt', 'test'); 
        roundSucces = true;
        if currentRound < length(train.sequence),
          fprintf('Press <RET> to advance to the next round\n');
          pause;        
        end
    end

    if currentRound == length(train.sequence) && roundSucces,
        currentSetting = 'Test';
        currentRound = 1;
        roundSucces = false;
        fprintf('Now going into TEST mode. Press <RET> to advance\n');
        pause;        
    end        
end
% % 
% % %% Exit script for running BBCI_BET stuff
% % if strcmp(currentSetting, 'RunAnalyzis'),
% %     fprintf('\n\nRun the bbci_bet routine in a second instance of Matlab\n');
% %     fprintf('When finished, set currentSetting = ''Test'' to resume the experiment\n\n');
% % end

%% Run the test trials
if strcmp(currentSetting, 'Test'),
     settings_bbci= {'bbci.start_marker', 252, ...
                  'bbci.quit_marker', 254, ...
                  'bbci.adaptation.running',0,...
                  'bbci.filt',[]};

      bbci_cfy= [TODAY_DIR '/adapted_cls.mat'];
      cfy = load(bbci_cfy);
      cfy_adapted = load([TODAY_DIR '/adapted_clsTmp.mat'], 'cls');
      cfy.cls = cfy_adapted.cls;
      save(bbci_cfy, '-STRUCT', 'cfy');
      
    if ~roundSucces,
        trials_to_do = [currentRound:length(test.sequence)];
    else
        trials_to_do = [currentRound+1:length(test.sequence)];
    end
    
    for currentRound = trials_to_do,
        roundSucces = false;
        fileName = test.files{test.sequence(currentRound)};
        bvr_startrecording(['impedanceBefore' VP_CODE]);
        pause(1);
        bvr_sendcommand('stoprecording');
        pause(1);
        try
            bvr_startrecording([fileName VP_CODE], 'impedances', 0); 
            pause(2);
        catch
            error('The BrainVision recorder is not responding!');
        end
        fprintf('Testing round %i: %s\n', currentRound, test.conditions{test.sequence(currentRound)});
        %%%%%%%%%%%%%%%%%%%% Start the actual experiments
%         send_xmlcmd_udp('fc-signal', 's:TODAY_DIR', TODAY_DIR, 's:VP_CODE', VP_CODE, 's:BASENAME', fbname);        
        send_xmlcmd_udp('interaction-signal', 's:_feedback', 'MultiScaleFeedback','command','sendinit');
        pause(5);
        fbOpts = fieldnames(fbopt);
        for i = 1:length(fbOpts),
            send_xmlcmd_udp('interaction-signal', fbOpts{i}, getfield(fbopt, fbOpts{i}));
            pause(.005);
        end
        if strcmp(test.conditions{test.sequence(currentRound)}, 'Single'),
            send_xmlcmd_udp('interaction-signal', 'i:classifier_limit', 1);
        else
            send_xmlcmd_udp('interaction-signal', 'i:classifier_limit', 0);
        end
        send_xmlcmd_udp('interaction-signal', 'i:number_of_trials', test.trials);        
        send_xmlcmd_udp('interaction-signal', 'colours', [colors(test.colSeq(currentRound,:))]); %% How to send this properly
        send_xmlcmd_udp('interaction-signal', 'command', 'play');        
        bbci_bet_apply(bbci_cfy, settings_bbci{:})
        send_xmlcmd_udp('interaction-signal', 'command', 'quit');        
        %%%%%%%%%%%%%%%%%%%% End the actual experiments        
        pause(1);
        bvr_sendcommand('stoprecording');
        save([TODAY_DIR 'FeedbackVars' test.conditions{test.sequence(currentRound)} int2str(currentRound) '.mat'], 'bbci', 'fbopt', 'test'); 
        roundSucces = true;
        fprintf('Press <RET> to advance to the next round\n');
        pause;
    end

    if currentRound == length(test.sequence) && roundSucces,
        currentSetting = 'Finished';
        pause;
    end     
end

%% Close up experiment
if strcmp(currentSetting, 'Finished'),
    send_xmlcmd_udp();
end
