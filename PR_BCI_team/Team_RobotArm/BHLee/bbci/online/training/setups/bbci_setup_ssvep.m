opt = struct('ival',[-150 800]);
opt.nPat = 2;
opt.usedPat = 1:2*opt.nPat;
opt.baseline = 150;
opt.clab = {'*'};
opt.model = 'RLDAshrink';
opt.dar_ival = [-150 800];
opt.dar_base = 150;
opt.editable = {'baseline','the baseline to use for classification';...
                'clab','Channels to use';...
                'selectival','ival length for classification';...
                'ival','Time interval regarding stimulus to use';...
                'jMeans','number of jumping means'};

opt.band = [];
opt.filtOrder = 5;
opt.ilen_apply = 1000;

opt.threshold = inf;
opt.nclassesrange = [2,2];
