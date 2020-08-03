function [cnt, mrk, sbjData] = load_calibration_data_T9(VPDir, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'fs', 200 ... % sampling rate
    ,'classDef', {11:19, 1:9; 'Target', 'Non-target'} ... 
);


sbj = strtok(VPDir, '_');
file = fullfile('/home/bbci/data/bbciRaw', VPDir, ...
    ['T9SpellerCalibration' sbj '*']);
[cnt, mrk_orig]= eegfile_readBV(file, 'fs', opt.fs);
mrk = mrk_defineClasses(mrk_orig, opt.classDef);
% get the classifier
tmp = load('/home/bbci/data/results/projects/T9Speller/allSbjData.mat');
sbjData = tmp.allSbjData.(sbj);
sbjData.setup_opts = sbjData.bbci.setup_opts;
sbjData.classifier = sbjData.classifier;
cnt = proc_selectChannels(cnt, sbjData.bbci.analyze.features.clab);
