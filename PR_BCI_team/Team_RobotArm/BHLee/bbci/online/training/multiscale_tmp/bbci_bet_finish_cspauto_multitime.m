
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls


% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, analyze.clab));
% $$$ if bbci.player==2
% $$$   cont_proc.clab= strcat('x', cont_proc.clab);
% $$$ end
for iival = 1:size(analyze.csp_a,1)
    cont_proc(iival).clab = clab;
    cont_proc(iival).procFunc = {'online_linearDerivation','online_filt'};
    cont_proc(iival).procParam = {{analyze.csp_w{iival}},{analyze.csp_b(iival,:),analyze.csp_a(iival,:)}};
%     cont_proc(iival).procFunc = {'online_linearDerivation'};
%     cont_proc(iival).procParam = {{analyze.csp_w{iival}}};


    feature(iival).cnt = iival;
    feature(iival).ilen_apply = opt.ival(iival,2)-opt.ival(iival,1);
    feature(iival).proc = {'proc_variance','proc_logarithm'};
    feature(iival).proc_param = {{},{}};

    cls(iival).fv = iival;
    cls(iival).applyFcn = getApplyFuncName(opt.model);
    cls(iival).C = trainClassifier(analyze.features{iival},opt.model);
    cls(iival).mrk_start={1,2};
end

bbci.filt=[];
