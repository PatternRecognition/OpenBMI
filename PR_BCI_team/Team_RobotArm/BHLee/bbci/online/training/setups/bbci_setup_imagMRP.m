opt = struct('ival',500:500:1000);
opt.sma = 2000;
opt.ma = 100;
opt.causal = 'causal';
opt.jMeans = 50;
opt.clab = {'F3-4','FFC#','FC#','CFC#','C#', ...
            'CCP#','CP#','PCP#','P5-6','O1,2'};

opt.threshold = inf;

opt.dar_ival = [-500 5500];
opt.dar_baseline = [-500 0];
opt.dar_scalps = 500:1000:3500;
opt.dar_sma = 2000;
opt.dar_ma = 100;
opt.dar_causal = 'centered';

opt.nclassesrange = [2,inf];
opt.model = 'LDA';


opt.editable = {'sma','subtract moving average';...
                'ma','moving average';...
                'clab','Channels to use';...
                'threshold','threshold to use for outlierness';...
                'ival','Time interval regarding stimulus to use';...
                'jMeans','number of jumping means'};
