function [cnt, mrk, sbjData] = load_data_HexoSpeller(VPDir, varargin)

% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt ...
%     );

sbj = strtok(VPDir, '_');
file_name = fullfile(VPDir, ['HexoSpellerVE_' sbj '_high_Hz']);
[cnt, mrk]= eegfile_loadMatlab(file_name);
mrk.className = {'Target', 'Non-target'};
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
s = load(fullfile('/home/bbci/data/bbciRaw', VPDir, ...
    ['bbci_classifier_HexoSpellerVE_' sbj '.mat']));
sbjData = [];
sbjData.classifier = s.cls;
sbjData.setup_opts = s.bbci.setup_opts;
sbjData.selectival = s.bbci.setup_opts.cfy_ival;
cnt = proc_selectChannels(cnt, s.bbci.analyze.features.clab);
