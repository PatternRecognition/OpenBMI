% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt,opt.clab));
cont_proc = struct('clab',{clab});
% $$$ if bbci.player==2
% $$$   for i = 1:length(cont_proc.clab)
% $$$     cont_proc.clab{i} = ['x',cont_proc.clab{i}];
% $$$   end
% $$$ end
cont_proc.procFunc = {'online_filt'};
cont_proc.procParam = {{analyze.emg_b,analyze.emg_a}};

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {'proc_bipolarEMG','proc_detectEMG'};
feature.proc_param = {{},{}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);




