
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls

% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, analyze.clab));
cont_proc = struct('clab',{clab});
cont_proc.procFunc = {'online_linearDerivation', ...
                      'online_filterbank'};
cont_proc.procParam = {{analyze.spat_w}, ...
                       {analyze.filt_b, analyze.filt_a}};

feature = struct('cnt',1);
feature.ilen_apply = opt.ilen_apply;
feature.proc = {'proc_variance','proc_logarithm'};
feature.proc_param = {{},{}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
fv= analyze.features;

if opt.allow_reselecting_laps,
  %% The classifier is only trained on the 3 selected Laplacian channels.
  %% In order to be able switch to other channels during the feedback
  %% (bbci_adaptation_...) features from all Laplacian channels are calculated.
  %% Here we insert components with 0 weight into the classifier to
  %% disregard the non-selected Laplacian channels.
  fv.x= fv.x(find(analyze.isactive),:);
  cls.C= trainClassifier(fv, opt.model);
  w_tmp= cls.C.w;
  cls.C.w= zeros(size(analyze.features.x,1), 1);
  cls.C.w(find(isactive))= w_tmp;
else
  cls.C= trainClassifier(fv, opt.model);
end

bbci.adaptation= copy_struct(bbci.setup_opts, ...
                             'nlaps_per_area', ...
                             'motorarea');
