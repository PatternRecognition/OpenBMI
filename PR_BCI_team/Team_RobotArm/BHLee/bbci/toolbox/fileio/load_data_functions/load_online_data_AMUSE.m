function [cnt, mrk, sbjData] = load_online_data_AMUSE(VPDir, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'include_mastoids', 0 ...
    );

sbj = strtok(VPDir, '_');
file_name = fullfile(VPDir, ['OnlineRunFile' sbj '_high_hz']);
[cnt mrk] = eegfile_loadMatlab(file_name, {'cnt', 'mrk'});
mrk = mrk_selectClasses(mrk, [2 1]);
mrk.className = {'Target', 'Non-target'};
% load the bbci file to get classifier stuff
s = load(fullfile('/home/bbci/data/bbciRaw', VPDir, 'bbci_classifier.mat'));
sbjData = [];
sbjData.setup_opts = s.bbci.setup_opts;
sbjData.classifier = s.cls;
sbjData.selectival = s.bbci.setup_opts.selectival;
select_clab = s.bbci.analyze.features.clab;
if opt.include_mastoids
    select_clab(end+[1,2]) = {'MasL', 'MasR'};
end
cnt = proc_selectChannels(cnt, select_clab);
if opt.include_mastoids
    % add linked mastoids
    A = [0.5; 0.5];
    cnt_mast = proc_selectChannels(cnt, {'MasL', 'MasR'});
    cnt_mast = proc_linearDerivation(cnt_mast, A);
    cnt_mast.clab = {'MasLink'};
    cnt = proc_appendChannels(cnt, cnt_mast);
end