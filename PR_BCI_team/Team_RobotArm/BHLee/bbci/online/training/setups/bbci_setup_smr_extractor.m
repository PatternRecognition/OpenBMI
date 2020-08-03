% All setups need to define a struct 'opt' with all required fields

opt.clab= {'not','E*','Fp*','AF*','OI*','I*','*9','*10'};
opt.clab= intersect(scalpChannels, Cnt.clab(chanind(Cnt, opt.clab)));
opt.band = 'auto';
opt.filtOrder = 5;
opt.ilen_apply = [750 7500];
opt.threshold = inf;
opt.nclassesrange = [1 1];
