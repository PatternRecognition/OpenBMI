function [err, err_std, err_train, err_train_std]= ...
    validate_method(cnt, mrk, method, tcl, verbose)
%[err, err_std, err_train, err_train_std]= ...
%      validate_method(cnt, mrk, method, tcl, <verbose=0>)
%
% IN:  cnt     - continuous data structure
%      mrk     - marker structure
%      verbose - verbosity level: 0 no output, 1 xVal output, 
%                                 2 xVal + selectModel output
%
% OUT: err     - cross-validation test error
%      err_std - standard error of the mean

% bb 06/03, ida.first.fhg.de

epo= proc_selectChannels(cnt, method.chans);
epo= makeEpochs(epo, mrk, [-method.ilen 0] + tcl);
eval(method.proc);

if isfield(method,'msTrials'),
  classy= selectModel(fv, method.model, method.msTrials, verbose-1);
  [ee, es]= doXvalidationPlus(fv, classy, method.xTrials, ...
                              max(1,verbose-1));
else
  [ee, es]= doXvalidationPlus(fv, method.model, method.xTrials, ...
                              max(1,verbose-1));
end

err= ee(1);
err_std= es(1);
err_train= ee(2);
err_train_std= es(2);
