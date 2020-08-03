
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt,opt.clab));
cont_proc = struct('clab',{clab});
% $$$ if bbci.player==2
% $$$   cont_proc.clab= strcat('x', cont_proc.clab);
% $$$ end
cont_proc.procFunc = {'proc_meanAcrossTime','online_filt'};
cont_proc.procParam = {{},{analyze.b,analyze.a}};

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {[],[]};
feature.proc_param = {{},{}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);

