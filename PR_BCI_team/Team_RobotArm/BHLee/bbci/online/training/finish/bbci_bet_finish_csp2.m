
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt,opt.clab));
cont_proc = struct('clab',{clab},'procFunc',{{'proc_linearDerivation','online_filterbank'}},'procParam',{{{hlp_w},{filt_b,filt_a}}});

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {'proc_variance','proc_logarithm'};
feature.proc_param = {{},{}};

cls = struct('fv',1:size(opt.band,1));
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);

