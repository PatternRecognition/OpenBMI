% All setups need to define a struct 'opt' with all required fields

clear opt

opt.clab = {'*'};
opt.ival = [];
opt.default_ival = [750 3500];
opt.band = [];
opt.filtOrder = 5;
[opt.filt_b, opt.filt_a]= ellip(10, 0.1, 80, 45*2/bbci.fs);
opt.proc_fs= 100;
opt.ilen_apply = 1000;
opt.dar_ival = [-500 5000];
opt.model = 'LSR';
opt.threshold = 'auto';
opt.editable = {'clab','Channels to use';...
                'ival','Time interval regarding stimulus to use';...
                'band','Frequency band to use';...
                'threshold','threshold to use for outlier removal'};
opt.nclassesrange = [2,2];
