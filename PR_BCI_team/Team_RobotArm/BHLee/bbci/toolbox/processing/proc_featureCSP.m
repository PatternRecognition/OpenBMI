function fv=proc_featureCSP(fv, W)
fv=proc_logarithm(proc_variance(proc_linearDerivation(fv, W)));
