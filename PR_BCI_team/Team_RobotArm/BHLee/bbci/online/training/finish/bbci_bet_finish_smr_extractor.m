% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

%% Processing of continuous signals:
%%  apply Laplace filter and band-pass filters for
%%  center frequency, lower and upper 'valley'
clab = Cnt.clab(chanind(Cnt, analyze.clab));
cont_proc = struct('clab',{clab});          

cont_proc.procFunc = {'online_linearDerivation', ...
                      'online_filterbank'};         
cont_proc.procParam = {{analyze.spat_w}, ...        
                       {analyze.filt_b, analyze.filt_a}};

%% Extraction of features: calculate log-variance of selected Laplacian
%%  channels in the three frequency bands (peak and valleys) and apply
%%  the 'SMR extractor' (subtract mean band power at valleys from
%%  band power at peak, and normalize to specified range).
%%  (1) short-term window
feature = struct('cnt',1);
feature.ilen_apply= opt.ilen_apply(1);
feature.proc = {'proc_variance','proc_logarithm', ['proc_' analyze.extractor_fcn]};
feature.proc_param = {{},{}, {analyze.opt_smr}};

%%  (2) long-term window
feature(2).cnt= 1;
feature(2).ilen_apply= opt.ilen_apply(2);
feature(2).proc = {'proc_variance','proc_logarithm', ['proc_' analyze.extractor_fcn]};
feature(2).proc_param = {{},{}, {analyze.opt_smr}};

%% do not use a classifier - the feature itself is used as feedback
cls(1).fv= 1;
cls(1).applyFcn = 'apply_nothing';
cls(1).C= [];

%% do not use a classifier - the feature itself is used as feedback
cls(2).fv= 2;
cls(2).applyFcn = 'apply_nothing';
cls(2).C= [];
