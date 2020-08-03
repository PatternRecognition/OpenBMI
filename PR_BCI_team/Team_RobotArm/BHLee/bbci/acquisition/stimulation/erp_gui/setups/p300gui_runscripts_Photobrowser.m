% Different runscripts that can be accessed through the GUI
global VP_SCREEN;
clear exp_opt;
clear exp_res;

if ~exist('simulate_run', 'var'),
    simulate_run = 0;
end

dropbox_root = 'E:\Dropbox\FSL_Study_Root\';
standard_root = 'E:\temp\Home\';
pb_db = 'E:\svn\p300_photobrowser\p300_database.sqlite';


record_filename = 'not_set';
% pyff_host = '127.0.0.1';
% pyff_port = 12345;

switch do_action,
    case 'give_available_experiments',
        available_experiments = {'Rest state EEG', 'Artifact measurement', 'Short rest state EEG', 'Calibration full', 'Calibration flash', 'Calibration mask', 'Standard task 1', 'Standard task 2', 'Standard task 3', 'Standard task 4', 'Standard task 5', 'Free mode'};
        requires_online = [0 0 0 0 0 1 1 1 1 1 1 1];
        feedback_type = 'pyff';
        available_parameters = {
            {}, ...
            {}, ...
            {}, ...            
            {'num_iterations', 12}, ...            
            {'num_iterations', 12}, ...
            {'num_iterations', 12}, ...
            {'num_iterations', 12}, ...
            {'num_iterations', 12}, ...
            {'num_iterations', 12}, ...
            {'num_iterations', 12}, ...
            {'num_iterations', 12}, ...            
            {'num_iterations', 12}, ...            
            };
        
    case 'Rest state EEG'
        record_filename = 'PhotoBrowser_rest';
        n_blocks = 3; % number of eyes open / closed blocks
        eyes_open_time = 60000; % time in ms
        eyes_closed_time = 60000; % time in ms
        use_signal_server = strcmp(func2str(acquire_func), 'acquire_sigserv'); % we assume that gTec is the only alternative to BV
        % sequence of actions for resting state measurement
        % P - pause
        % Fx - see setup_artifacts_and_resting_measurement.m for the codes
        % R[x](some_stuff) - repeat some_suff x times
        % markers are sent for each captial-letter-action
        seq = ['P2000 f21P2000 ' ...
            sprintf('R[%d](F15P%d F14P%d) ', n_blocks, eyes_open_time, eyes_closed_time)...
            'F20P1000'];
        [seq, wav, opt]= setup_artifacts_and_resting_measurement('language', 'italian', ...
            'seq', seq, 'show_description', 0);
        stim_artifactMeasurement(seq, wav, opt, ...
            'filename', record_filename, ...
            'useSignalServer', use_signal_server, ...
            'test', simulate_run);
    
    case 'Short rest state EEG'
        record_filename = 'PhotoBrowser_rest_short';
        n_blocks = 1; % number of eyes open / closed blocks
        eyes_open_time = 60000; % time in ms
        eyes_closed_time = 60000; % time in ms
        use_signal_server = strcmp(func2str(acquire_func), 'acquire_sigserv'); % we assume that gTec is the only alternative to BV
        % sequence of actions for resting state measurement
        % P - pause
        % Fx - see setup_artifacts_and_resting_measurement.m for the codes
        % R[x](some_stuff) - repeat some_suff x times
        % markers are sent for each captial-letter-action
        seq = ['P2000 f21P2000 ' ...
            sprintf('R[%d](F15P%d F14P%d) ', n_blocks, eyes_open_time, eyes_closed_time)...
            'F20P1000'];
        [seq, wav, opt]= setup_artifacts_and_resting_measurement('language', 'italian', ...
            'seq', seq, 'show_description', 0);
        stim_artifactMeasurement(seq, wav, opt, ...
            'filename', record_filename, ...
            'useSignalServer', use_signal_server, ...
            'test', simulate_run);    
        
    case 'Artifact measurement'
        record_filename = 'PhotoBrowser_artifact';   
        n_blocks = 6; % one block contains all directions once
        time_for_movement = 4500; % time in ms
        pause_after_n_blocks = 3; % if smaller than 1 then no pause will be made, 
            % otherwise this is the number of blocks after which an extra pause comes
        pause_time = 10000; % time in ms
        use_signal_server = strcmp(func2str(acquire_func), 'acquire_sigserv'); % we assume that gTec is the only alternative to BV
        % create sequence of actions 
        % P - pause
        % Fx - see setup_artifacts_and_resting_measurement.m for the codes
        % R[x](some_stuff) - repeat some_suff x times
        % markers are sent for each captial-letter-action
        seq = sprintf('P2000 F15P3000 f6f7P%d', time_for_movement); % start with a short pause and eyes_open command
        directions = {'F8', 'F9', 'F10', 'F11'};
        n_directions = length(directions);
        for b=1:n_blocks
            idx = randperm(n_directions);
%             seq = sprintf('%s -block%d- ', seq, b); %% DEBUG
            for k=1:n_directions
                % add the direction to the sequence
                seq = sprintf('%s%sP%d ', seq, directions{idx(k)}, time_for_movement);
                % add 'middle' to the sequence
                seq = sprintf('%sF7P%d ', seq, time_for_movement);                
            end
            if pause_after_n_blocks && mod(b, pause_after_n_blocks)==0 && b<n_blocks
                seq = sprintf('%sf21P%d f6f7P%d', seq, pause_time, time_for_movement);
            end
        end
        seq = sprintf('%sP2000 F20P1000', seq);
        [seq, wav, opt]= setup_artifacts_and_resting_measurement('language', 'italian', ...
            'seq', seq, 'show_description', 0);
        stim_artifactMeasurement(seq, wav, opt, ...
            'filename', record_filename, ...
            'useSignalServer', use_signal_server, ...
            'test', simulate_run);
        
    otherwise
        pyff('init', 'p300_photobrowser');  
        try, rmdir(standard_root, 's'); end;
        copyfile([dropbox_root 'Training\Training_task'], standard_root);
        clear exp_opt stored_set;
        stp_file = [dropbox_root 'settings\' VP_CODE '_pb_setup.m'];        
        if exist(stp_file, 'file'), 
            run(stp_file)
        else
            stored_set.file_suffix = 'full';
        end
        exp_opt = set_defaults(gui_set_opts, stored_set);
        exp_opt = set_defaults(exp_opt, ...
            'logging_directory', [TODAY_DIR 'pblog\'], ...
            'screen_x', -1280, ...
            'screen_y', 0, ...
            'screen_width', 1280, ...
            'screen_height', 1024, ...
            'num_blocks', 2, ...
            'num_subtrials_per_iteration', 2, ...
            'num_trials', 4, ...
            'cue_presentation_duration', 5000, ...
            'inter_block_countdown_duration', 3000, ...
            'inter_block_pause_duration', 8000, ...
            'inter_stimulus_duration', 100, ...
            'inter_trial_duration', 3000, ...
            'post_cue_presentation_pause_duration', 1000, ...
            'post_subtrail_pause_duration', 1000, ....
            'pre_cue_pause_duration', 1000, ...
            'result_presentation_duration', 2000, ...
            'stimulus_duration', 100, ...
            'display_selection_enable', true, ...
            'mask_sequence_enable', true, ...
            'online_mode_enable', true, ...
            'row_col_eval_enable', true, ...
            'show_empty_image_frames_enable', false, ...
            'startup_sleep_duration', 10000, ...
            'mask_enable', true, ...
            'flash_enable', true, ...
            'rotation_enable', true, ...
            'scaling_enable', true, ...
            'testing_mode', false, ...
            'personal_directory', '__ROOT__/Condiviso/');

        switch do_action
            case 'Calibration full',
                exp_opt.copy_task = true;
                exp_opt.online_mode_enable = false;
                exp_opt.root_directory = standard_root;
                record_filename = 'PhotoBrowser_train_full_';
                exp_opt.mask_enable = true;
                exp_opt.rotation_enable = true;
                exp_opt.scaling_enable = true;                
                exp_opt.flash_enable = true;
                
            case 'Calibration mask',
                exp_opt.copy_task = true;
                exp_opt.online_mode_enable = false;
                exp_opt.flash_enable = false;
                exp_opt.mask_enable = true;                
                exp_opt.scaling_enable = false;
                exp_opt.rotation_enable = false;
                exp_opt.root_directory = standard_root;      
                record_filename = 'PhotoBrowser_train_mask_';
                
            case 'Calibration flash'
                exp_opt.copy_task = true;
                exp_opt.online_mode_enable = false;
                exp_opt.flash_enable = true;
                exp_opt.scaling_enable = false;
                exp_opt.rotation_enable = false;
                exp_opt.mask_enable = false;
                exp_opt.root_directory = standard_root;      
                record_filename = 'PhotoBrowser_train_flash_';                
                
            case {'Standard task 1', 'Standard task 2', 'Standard task 3', 'Standard task 4', 'Standard task 5'},
                blk_num = [2 2 3 5 5];
                task_num = str2num(do_action(end));
                exp_opt.num_blocks = blk_num(task_num);
                exp_opt.num_trials = 3;                
                record_filename = ['PhotoBrowser_standard_' do_action(end) '_' exp_opt.file_suffix '_'];
                exp_opt.root_directory = standard_root;
                
            case 'Free mode',
                record_filename = ['PhotoBrowser_free_mode_' exp_opt.file_suffix '_'];
                exp_opt.num_blocks = 20;
                exp_opt.root_directory = [dropbox_root VP_CODE '\'];
               
        end
        
        if simulate_run,
            exp_opt.testing_mode = true;
        end
              
        %start the actual stuff
        pause(2);
        fbOpts = fieldnames(exp_opt);
        for optId = 1:length(fbOpts),
            if isnumeric(getfield(exp_opt, fbOpts{optId})),
                send_xmlcmd_udp('interaction-signal', ['i:' fbOpts{optId}], getfield(exp_opt, fbOpts{optId}));
            elseif ischar(getfield(exp_opt, fbOpts{optId})),
                send_xmlcmd_udp('interaction-signal', ['s:' fbOpts{optId}], getfield(exp_opt, fbOpts{optId}));
            else
                send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(exp_opt, fbOpts{optId}));
            end
            pause(.005);
        end        
        pyff('play');

        if ~simulate_run,
            clear rec_opt;
            if strcmp(func2str(acquire_func), 'acquire_bv'),
                % this is easy
                rec_opt.impedances = 0;
                rec_opt.append_VP_CODE = 1;
                bvr_startrecording(record_filename, rec_opt);
            else
                % gTec sucks bigtime!
                signalServer_startrecoding([record_filename VP_CODE]);
            end
        end
end