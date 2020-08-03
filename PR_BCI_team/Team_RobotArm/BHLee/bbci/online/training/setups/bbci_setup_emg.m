% All setups need to define a struct 'opt' with all required fields

opt.clab = {'EM*'};
opt.ival = [200 1000];
opt.ilen_apply = 200;
opt.model = 'LDA';
opt.editable = {'clab','Channels to use';...
                'ival','Time interval regarding stimulus'};

opt.nclassesrange = [2,inf];
