% All setups need to define a struct 'opt' with all required fields

opt.nPat = 2;
opt.usedPat = 1:2*opt.nPat;
opt.clab = {'not','E*','Fp*','AF*','I*'};
opt.ival = [];
opt.default_ival = [3000 8000];
opt.filtOrder = 2;
opt.band=[0.01 0.2];
opt.ilen_apply = 1000;
opt.dar_ival = [-5000 15000];
opt.model = 'LDA';
opt.threshold = inf;
opt.editable = {'nPat','Number of pattern to use for each class';...
                'usedPat','Specified patterns to use';...
                'clab','Channels to use';...
                'ival','Time interval regarding stimulus to use';...
                'band','Frequency band to use';...
                'threshold','threshold to use for outlier removal'};
opt.nclassesrange = [2,2];
