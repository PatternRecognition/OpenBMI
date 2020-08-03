function [cnt, mrk, train_epos, test_epos, sbjData] = load_data_T9wharp(VPDir, varargin)

trial_per_block = 9;
n_blocks_calibration = 3;

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'n_calibration_trials', trial_per_block*n_blocks_calibration ...
    ,'include_mastoids', 0 ...
    );

% load the complete data set
sbj = strtok(VPDir, '_');
file = fullfile('/home/bbci/data/bbciMat', VPDir, ...
    ['T9wharp_' sbj '_cnt_mrk_high_Hz.mat']);
s = load(file);
cnt = s.cnt;
select_clab = scalpChannels;
if opt.include_mastoids
    select_clab(end+[1,2]) = {'MastL', 'MastR'};
end
cnt = proc_selectChannels(cnt, select_clab);
% rename the mastoid channels to be conform with AMUSE
cnt.clab([end-1,end]) = {'MasL', 'MasR'};
if opt.include_mastoids
    % add linked mastoids
    A = [0.5; 0.5];
    cnt_mast = proc_selectChannels(cnt, {'MasL', 'MasR'});
    cnt_mast = proc_linearDerivation(cnt_mast, A);
    cnt_mast.clab = {'MasLink'};
    cnt = proc_appendChannels(cnt, cnt_mast);
end

file = fullfile('/home/bbci/data/bbciMat', VPDir, ...
    ['T9wharp_' sbj '_mrkAdvanced.mat']);
s2 = load(file);
mrk = s2.mrk;
mrk = mrk_selectClasses(mrk, [1,2]);

% separate epochs in 'calibration' and 'test'
trial_wait_time = 10000; % time between trials (stim sequences) in ms
trial_wait_time_samples = mrk.fs*trial_wait_time/1000;
trial_start_idx = [1, 1+find(diff(mrk.pos) > trial_wait_time_samples)];
trial_end_idx = [trial_start_idx(2:end)-1, length(mrk.pos)];

train_epos = 1:trial_end_idx(opt.n_calibration_trials);
test_epos = (train_epos(end)+1):length(mrk.pos);


% create sbjData with selectival info
sbjData = [];
sbjData.setup_opts = []; % not available
sbjData.classifier = []; % not available
global BCI_DIR
file = fullfile(BCI_DIR, 'investigation/projects/tobi/student_project_T9wharp', 'ivals.mat');
dat = load(file, 'ivals_struct');
sbjData.selectival = dat.ivals_struct.(sbj);

