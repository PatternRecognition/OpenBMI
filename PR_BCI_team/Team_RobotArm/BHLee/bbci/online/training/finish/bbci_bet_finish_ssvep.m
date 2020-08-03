
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
% clab = Cnt.clab(chanind(Cnt,opt.clab));
epo = proc_selectChannels(epo, 'not', 'E*');
cont_proc = struct('clab',{epo.clab});
cont_proc.procFunc = {'online_filterbank','proc_linearDerivationSSVEP'};
cont_proc.procParam = {{analyze.csp_b,analyze.csp_a},{analyze.freq_matrix, analyze.csp_w}};

marker_output = struct();
marker_output.marker = {1 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18};
marker_output.value = [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8];
marker_output.no_marker = 0;

feature = struct('cnt',1);
feature.ilen_apply = 300; %10 to account for difference in marker position interpretation

feature.proc = {'proc_variance','proc_logarithm'};
feature.proc_param = {{},{}};

cls = struct('fv',1);
%cls.applyFcn = getApplyFuncName(opt.model);
cls.applyFcn ='apply_separatingHyperplaneSSVEP'
for i=1:length(analyze.features)
cls.C{i} = trainClassifier(analyze.features{i},opt.model);
end
