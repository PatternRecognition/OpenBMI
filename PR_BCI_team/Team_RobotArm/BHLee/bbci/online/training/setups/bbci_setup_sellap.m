% All setups need to define a struct 'opt' with all required fields

opt.clab = {'not','E*','Fp*','AF*','OI*','I*','*9','*10'};
opt.ival = [];
opt.default_ival = [1250 3500];
opt.band = [];
opt.filtOrder = 5;
opt.ilen_apply = 750;
opt.visu_ival = [-500 5000];
opt.model = {'RLDAshrink', 'scaling',1};
opt.threshold = inf;
opt.nclassesrange = [2,2];
