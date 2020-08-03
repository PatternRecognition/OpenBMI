
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls


% Extract channel labels clab
clab = Cnt.clab(chanind(Cnt, bbci.analyze.clab));
bbci.cont_proc = struct('clab',{clab});
% $$$ if bbci.player==2
% $$$   cont_proc.clab= strcat('x', cont_proc.clab);
% $$$ end

bbci.cont_proc.proc= {{@online_linearDerivation, bbci.analyze.spat_w, bbci.analyze.features.clab},{@online_filt, bbci.analyze.filt_b, bbci.analyze.filt_a}};
bbci.feature.proc= {{@proc_variance},{@proc_logarithm}};
bbci.feature.ival= [-750 0];

bbci.classifier.C= trainClassifier(analyze.features, opt.model);
