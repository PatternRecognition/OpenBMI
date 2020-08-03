%% load_online_data
function [cnt, mrk, sbjData] = load_online_data(params)
% load online data for several experiments

% default parameter settings
params = set_defaults(params ...
    ,'sampling_rate', 250 ...
    ,'low_pass_cutoff_freqs', [80, 95] ...
);




if strcmp(params.experiment_paradigm, 'T9')
    s = load(fullfile('/home/bbci/data/bbciMat', params.VPDir, 'OnlineData_cnt_mrk_250Hz.mat')); % cnt mart_target and mark_chosen
%     s = load(fullfile('/home/bbci/data/bbciMat', params.VPDir, 'OnlineData_cnt_mrk.mat')); % cnt mart_target and mark_chosen
    cnt = s.cnt;
    s.cnt = [];
    mrk = s.mrk_target;
    % get the classifier
    tmp = load('/home/bbci/data/results/projects/T9Speller/allSbjData.mat');
    sbjData = tmp.allSbjData.(params.sbj);
    sbjData.setup_opts = sbjData.bbci.setup_opts;
    sbjData.classifier = sbjData.classifier;
    cnt = proc_selectChannels(cnt, sbjData.bbci.analyze.features.clab);
elseif strcmp(params.experiment_paradigm, 'AMUSE')
    file_name = fullfile(params.VPDir, ['OnlineRunFile' params.sbj 'high_hz']);
    [cnt mrk] = eegfile_loadMatlab(file_name, {'cnt', 'mrk'});
    mrk = mrk_selectClasses(mrk, [2 1]);
    mrk.className = params.classDef(2,:);
    % load the bbci file to get classifier stuff
    s = load(fullfile('/home/bbci/data/bbciRaw', params.VPDir, 'bbci_classifier.mat'));
    sbjData = [];
    sbjData.setup_opts = s.bbci.setup_opts;
    sbjData.classifier = s.cls;
    sbjData.selectival = s.bbci.setup_opts.selectival;
    cnt = proc_selectChannels(cnt, s.bbci.analyze.features.clab);
elseif strcmp(params.experiment_paradigm, 'onlineVisualSpeller_HexoSpeller')
    file_name = fullfile(params.VPDir, ['HexoSpellerVE_' params.sbj '_high_Hz']);
    [cnt, mrk]= eegfile_loadMatlab(file_name);
    % select only the online spelling phase data
    online_mode_indices = find(sum(mrk.mode(2:3,:)));
    mrk = mrk_selectEvents(mrk, online_mode_indices);
    mrk.className = params.classDef(2,:);
    mrk.pos= mrk.pos + 40/1000*mrk.fs;  % correct for delay of visual present.
    % markers in this study were 11 to 16 for first level decision and 21
    % to 26 for second level decision. Target markers are not different
    % from non-target markers. Bring that into the same format as for the
    % auditory paradigms, i.e. non-target markers smaller are <= 10 and the
    % corresponding target markers are non-target+10
    mrk.toe = mrk.toe - 10; % markers are now in the between 1 and 20
    mrk.toe(mrk.toe>10) = mrk.toe(mrk.toe>10) - 10; % now between 1 and 10
    mrk.toe(logical(mrk.y(1,:))) = mrk.toe(logical(mrk.y(1,:))) + 10; % target markers > 10, non-target < 10
    % load the classifier
    s = load(fullfile('/home/bbci/data/bbciRaw', params.VPDir, ...
        ['bbci_classifier_HexoSpellerVE_' params.sbj '.mat']));
    sbjData = [];
    sbjData.classifier = s.cls;
    sbjData.setup_opts = s.bbci.setup_opts;
    sbjData.selectival = s.bbci.setup_opts.cfy_ival;
    cnt = proc_selectChannels(cnt, s.bbci.analyze.features.clab);
else
    error('Unknown experiment paradigm!')
end

if params.do_artifact_rejection
    [mrk, rclab, rtrials] = reject_varEventsAndChannels(cnt, mrk, [0 800], 'visualize', 0, 'do_multipass', 1, 'do_bandpass', 0, 'whiskerlength', 2.5);
    fprintf('%i trials were removed with artifact rejection\n', length(rtrials));
    params.n_removed_trials = length(rtrials);
else
    params.n_removed_trials = 0;
end
