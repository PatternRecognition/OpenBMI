%% Runscript for the first pretty cool and truely multimodal BCI study by Xing-Wei and
%% Johannes


%% initialization

%init udp and pyff
bvr_sendcommand('viewsignals');
pause(2)
send_xmlcmd_udp('init', '127.0.0.1', 12345);
disp('parameters set')

%% -------- prepare Cap  -------
%% play sounds or show visual stimuli... ?!?


%% save impedances
bvr_startrecording('impedances', 'impedances', 1);
pause(1)
bvr_sendcommand('stoprecording');

%% main recording - setup
nTrials_perCondition = [12 12 30 30 36 36]; %has to be a multiple of 6
if sum(mod(nTrials_perCondition, 6)) > 0
    error('nTrials_perCondition has to consists of numbers that are multiple of 6')
end
if (nTrials_perCondition(1) ~= nTrials_perCondition(2) ) | (nTrials_perCondition(3) ~= nTrials_perCondition(4) ) | (nTrials_perCondition(5) ~= nTrials_perCondition(6) )
    error('nTrials must be the same for 1-2, 3-4 and 5-6')
end

nTargetsEach = nTrials_perCondition/6;

tmp_targetContainer = {};
for ii=1:4
    tmp_targetContainer{ii} = [];
    for jj = 1:nTargetsEach(ii)
        tmp_targetContainer{ii} = [tmp_targetContainer{ii} randperm(6)];
    end
end

tmp_targetContainer{5} = randperm(36);
tmp_targetContainer{6} = randperm(36);

% tmp_targetContainer contains a random sequence of targets for each
% condition

targetContainer = {[] [] []};
condition_list = {};
count = 0;
for ii = [1 3 5]
    count = count+1;
    dummy = [];
    for i_trial = 1: nTrials_perCondition(ii)
        my_randOrder = randperm(2)+ii-1;
        dummy = [dummy (my_randOrder)];
        targetContainer{count} = [targetContainer{count} ...
            tmp_targetContainer{my_randOrder(1)}(i_trial) tmp_targetContainer{my_randOrder(2)}(i_trial)];
    end
    condition_list = [condition_list dummy];
end


fname = [TODAY_DIR 'block_info.mat'];
if ~exist(fname, 'file')
    save(fname, 'targetContainer', 'condition_list')
    disp('saved block_info into file!')
else
    clear targetContainer condition_list
    load(fname)
    disp('loaded block_info from file!')
end

%% singleModality

fb_name = 'Center_Oddball'
send_xmlcmd_udp('interaction-signal', 's:_feedback', fb_name,'command','sendinit');
fname = ['audiVisual_cond_01_02_' VP_CODE]

desired_target = [];
desired_cond = [];
for ii = 1:4:length(condition_list(1))
    dum_inp = '';
    while not(strcmp(dum_inp, 'start'))
        dum_inp = input(sprintf('type ''start'' to initiate Trial %i - %i : ', ii, ii+4), 's');
    end
    bvr_startrecording(fname, 'impedances', 0);
    sprintf('starting the recording!')
    pause(2)
    desired_cond = condition_list{1}(ii:(ii+4));
    desired_target = targetContainer{1}(ii:(ii+4));
    run_trials(desired_target, desired_cond);
end


%% audioVisualDependent

fb_name = 'Center_Oddball'
send_xmlcmd_udp('interaction-signal', 's:_feedback', fb_name,'command','sendinit');
fname = ['audiVisual_cond_03_04_' VP_CODE]

desired_target = [];
desired_cond = [];
for ii = 1:4:length(condition_list(2))
    dum_inp = '';
    while not(strcmp(dum_inp, 'start'))
        dum_inp = input(sprintf('type ''start'' to initiate Trial %i - %i : ', ii, ii+4), 's');
    end
    bvr_startrecording(fname, 'impedances', 0);
    sprintf('starting the recording!')
    pause(2)
    desired_cond = condition_list{2}(ii:(ii+4));
    desired_target = targetContainer{2}(ii:(ii+4));
    run_trials(desired_target, desired_cond);
end




%% audioVisualIndependent
fb_name = 'TM_CenterSpeller'
send_xmlcmd_udp('interaction-signal', 's:_feedback', fb_name,'command','sendinit');

desired_target = [];
desired_cond = [];
for ii = 1:4:length(condition_list(3))
    dum_inp = '';
    while not(strcmp(dum_inp, 'start'))
        dum_inp = input(sprintf('type ''start'' to initiate Trial %i - %i : ', ii, ii+4), 's');
    end
    bvr_startrecording(fname, 'impedances', 0);
    sprintf('starting the recording!')
    pause(2)
    desired_cond = condition_list{3}(ii:(ii+4));
    desired_target = targetContainer{3}(ii:(ii+4));
    run_trials(desired_target, desired_cond);
end

