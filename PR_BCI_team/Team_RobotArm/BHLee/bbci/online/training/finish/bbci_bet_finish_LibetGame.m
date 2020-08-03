% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, analyze.features.clab));
cont_proc = struct('clab',{clab});

os= 1000/bbci.fs;  % duration of one sample in msec
feature = struct('cnt',1);
feature.ilen_apply = analyze.ival(end) - analyze.ref_ival(1) + os; 

feature.proc= {'proc_baseline', ...
               'proc_jumpingMeans'};
feature.proc_param= {{diff(analyze.ref_ival), 'beginning_exact'}, ...
                     {analyze.ival-analyze.ival(end)}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features, opt.model);
