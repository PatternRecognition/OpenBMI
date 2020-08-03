function [cnt, mrk, sbjData] = load_online_data_T9(VPDir, varargin)

% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt ...
%     );

sbj = strtok(VPDir, '_');
s = load(fullfile('/home/bbci/data/bbciMat', VPDir, 'OnlineData_cnt_mrk_250Hz.mat')); % cnt mart_target and mark_chosen
%     s = load(fullfile('/home/bbci/data/bbciMat', params.VPDir, 'OnlineData_cnt_mrk.mat')); % cnt mart_target and mark_chosen
cnt = s.cnt;
s.cnt = [];
mrk = s.mrk_target;
% get the classifier
tmp = load('/home/bbci/data/results/projects/T9Speller/allSbjData.mat');
sbjData = tmp.allSbjData.(sbj);
sbjData.setup_opts = sbjData.bbci.setup_opts;
sbjData.classifier = sbjData.classifier;
cnt = proc_selectChannels(cnt, sbjData.bbci.analyze.features.clab);
