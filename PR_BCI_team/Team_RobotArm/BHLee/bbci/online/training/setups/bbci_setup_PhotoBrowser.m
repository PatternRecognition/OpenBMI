opt = struct('ival',[-150 1000]);
opt.baseline = 150;

opt.clab = {'*'};
% opt.jMeans = 3;
% opt.model = {'boundErrorOfType1', 0.02, 'FisherDiscriminant'};
opt.model = 'FDshrink';

opt.dar_ival = [-150 1000];
opt.dar_base = 150;
% opt.dar_scalps = 0:50:400;

opt.editable = {'baseline','the baseline to use for classification';...
                'clab','Channels to use';...
                'selectival','ival length for classification';...
                'ival','Time interval regarding stimulus to use';...
                'jMeans','number of jumping means'};

% opt.nclassesrange = [2,inf];
