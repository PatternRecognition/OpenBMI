% All setups need to define a struct 'opt' with all required fields

opt.nPat = 3;
opt.usedPat = 'auto';
opt.clab = {'not','E*','Fp*','AF*','OI*','I*','*9','*10'};
opt.ival = [-500 5000];
opt.default_ival = [1250 3500];
opt.band = [0.1 6];
opt.filtOrder = 5;
opt.ilen_apply = 750;
opt.visu_ival = [-500 5000];
opt.visu_band = [0.1 10];
opt.baseline = [-150 100];
opt.model = 'RLDAshrink';
opt.threshold = inf;
opt.editable = {'nPat','Number of pattern to use for each class';...
                'usedPat','Specified patterns to use';...
                'clab','Channels to use';...
                'ival','Time interval regarding stimulus to use';...
                'band','Frequency band to use';...
                'threshold','threshold to use for outlier removal'};
opt.nclassesrange = [2,2];


opt.reject_opts = {'do_multipass' , 1 , 'whiskerperc' , 10, 'whiskerlength', 3 , 'do_relvar' , 1 , 'do_bandpass', 0};