% All setups need to define a struct 'opt' with all required fields

opt.nPat = 3;
opt.usedPat = 'auto';
opt.clab = {{'not','E*','Fp*','AF*','OI*','I*','*9','*10'}, ...
    {'not','E*','Fp*','AF*','OI*','I*','*9','*10'}};%CSP/LRP
% opt.ival = {[1000 4000], [-500 5000]};
opt.ival = {'auto', 'auto'};

opt.default_ival = {[2000 3000], [1250 3500]};%CSP/LRP
opt.band = {[13 15], [0.1 6]};%CSP/LRP (LRP is not really used ?!?)
opt.filtOrder = 5; %CSP
opt.ilen_apply = {750 750}; %CSP/LRP
opt.visu_ival = [-500 4000];%CSP
opt.visu_band = [0.1 25]; % CSP
opt.baseline = [-250 100]; % LRP
opt.model = {{'RLDAshrink', 'scaling',1, 'store_means',1, 'store_invcov',1,'store_extinvcov',1} {'RLDAshrink', 'scaling',1, 'store_means',1, 'store_invcov',1,'store_extinvcov',1}}; %CSP/LRP
opt.threshold = inf; %??
opt.editable = {'nPat','Number of pattern to use for each class';...
                'usedPat','Specified patterns to use';...
                'clab','Channels to use';...
                'ival','Time interval regarding stimulus to use';...
                'band','Frequency band to use';...
                'threshold','threshold to use for outlier removal'};
opt.nclassesrange = [2,2];


opt.reject_opts = {'do_multipass' , 1 , 'whiskerperc' , 10, 'whiskerlength', 3 , 'do_relvar' , 1 , 'do_bandpass', 0};