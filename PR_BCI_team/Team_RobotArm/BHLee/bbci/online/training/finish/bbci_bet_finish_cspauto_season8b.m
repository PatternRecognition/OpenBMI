
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls


% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, analyze.clab));
cont_proc = struct('clab',{clab});
% $$$ if bbci.player==2
% $$$   cont_proc.clab= strcat('x', cont_proc.clab);
% $$$ end
cont_proc(1).procFunc = {'online_linearDerivation','online_filt'};
cont_proc(1).procParam = {{analyze.csp_w},{analyze.csp_b,analyze.csp_a}};

feature(1).cnt = 1;
feature(1).ilen_apply = opt.ilen_apply;
feature(1).proc = {'proc_variance','proc_logarithm'};
feature(1).proc_param = {{},{}};

cls(1).fv = 1;
cls(1).applyFcn = getApplyFuncName(opt.model);
cls(1).C = trainClassifier(analyze.features,opt.model);

[requ_clab, analyze.spat_w]= ...
    getClabForLaplacian(strukt('clab',Cnt.clab), {'C3', 'C4'}, 'require_complete_neighborhood', 0);

cont_proc(2).clab = requ_clab;
cont_proc(2).procFunc = {'online_linearDerivation','online_filt'};
cont_proc(2).procParam = {{analyze.spat_w},{analyze.csp_b,analyze.csp_a}};

feature(2).cnt = 2;
feature(2).ilen_apply = opt.ilen_apply;
feature(2).proc = {'proc_variance','proc_logarithm', 'proc_mulevel'};
feature(2).proc_param = {{},{}, {mu}};

cls(2).fv = 2;
cls(2).applyFcn = getApplyFuncName(opt.model);
cls(2).C.w = [1; 0];
cls(2).C.b = 0;

cls(3).fv = 2;
cls(3).applyFcn = getApplyFuncName(opt.model);
cls(3).C.w = [0; 1];
cls(3).C.b = 0;
