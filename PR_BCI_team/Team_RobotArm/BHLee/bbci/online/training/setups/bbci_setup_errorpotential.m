opt = struct('ival',[-50 350]);
opt.baseline = 50;
opt.selectival = 300;
opt.clab = {'FC3,z,4','C5-6','CCP5-6','CP5-6','P3,z,4'};
opt.jMeans = 3;
opt.model = {'boundErrorOfType1', 0.02, 'FisherDiscriminant'};

opt.dar_ival = [-100 1000];
opt.dar_base = [-100 -50];
opt.dar_scalps = 0:50:400;

opt.editable = {'baseline','the baseline to use for classification';...
                'clab','Channels to use';...
                'selectival','ival length for classification';...
                'ival','Time interval regarding stimulus to use';...
                'jMeans','number of jumping means'};

opt.nclassesrange = [2,inf];
