% bbci_bet_finish_cspproto script
%    
% NOTE: this requires to generate an epo-struct first and store it under
% [DATA_DIR 'results/csp_paramspace_online/']. (see load_data_files.m in
% BCI_DIR/kraulems_analysis/csp_paramspace_online/)

% kraulem 10/06

% Extract channel labels clab
cont_proc = struct('clab',{analyze.clab});
cont_proc.procFunc = {'proc_linearDerivation','online_filt'};
cont_proc.procParam = {{analyze.csp_w},{analyze.csp_b,analyze.csp_a}};

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {'proc_variance','proc_logarithm'};
feature.proc_param = {{},{}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
cls.C = trainClassifier(analyze.features,opt.model);
