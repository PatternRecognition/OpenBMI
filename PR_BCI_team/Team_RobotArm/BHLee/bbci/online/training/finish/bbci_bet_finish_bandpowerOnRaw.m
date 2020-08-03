
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt,opt.clab));
cont_proc = struct('clab',{clab});
% $$$ if bbci.player==2
% $$$   cont_proc.clab= strcat('x', cont_proc.clab);
% $$$ end

if isempty(analyze.filt_b),
  cont_proc.procFunc = {'online_filt'};
  cont_proc.procParam = {{analyze.csp_b,analyze.csp_a}};
else
  cont_proc.procFunc = {'online_filt','online_filt'};
  cont_proc.procParam = {{analyze.filt_b_b,analyze.filt_a}, ...
                         {analyze.csp_b,analyze.csp_a}};
end

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
if analyze.fs==analyze.proc_fs,
  feature.proc = {'proc_variance','proc_logarithm'};
  feature.proc_param = {{},{}};
else
  feature.proc = {'proc_subsampleByLag','proc_variance','proc_logarithm'};
  feature.proc_param = {{10},{},{}};
end
cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);

