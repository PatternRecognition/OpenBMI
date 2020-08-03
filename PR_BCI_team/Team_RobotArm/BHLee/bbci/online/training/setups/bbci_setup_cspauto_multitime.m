% All setups need to define a struct 'opt' with all required fields

opt.nPat = 3;
opt.usedPat = 'auto';
opt.clab = {'not', 'Fp*'};
% opt.clab = {'not','E*','Fp*','AF*','OI*','I*','*9','*10'};
% opt.clab = {'C5-6','CCP5-6','CFC5-6','FC3', 'CP3', 'FC4', 'CP4'};
opt.ival = [];
opt.selival_opt = [500 1000 2000 2750];
opt.default_ival = [1250 3500];
opt.band = [];
opt.filtOrder = 5;
opt.ilen_apply = 750;
opt.visu_ival = [-500 5000];
opt.enlarge_ival_append = 'end';
opt.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1, 'store_extinvcov',1};
opt.threshold = inf;
opt.editable = {'nPat','Number of pattern to use for each class';...
                'usedPat','Specified patterns to use';...
                'clab','Channels to use';...
                'ival','Time interval regarding stimulus to use';...
                'band','Frequency band to use';...
                'threshold','threshold to use for outlier removal'};
opt.nclassesrange = [2,2];
