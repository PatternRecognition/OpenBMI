function [err, err_std, err_train, err_train_std]= ...
    validate_dtct_method(cnt, mrk, method, tcl, verbose)
%[err, err_std, err_train, err_train_std]= ....
%      validate_dtct_method(cnt, mrk, method, tcl, <verbose=0>)
%
% IN:  cnt     - continuous data structure
%      mrk     - marker structure
%      verbose - verbosity level: 0 no output, 1 xVal output, 
%                                 2 xVal + selectModel output
%
%
% OUT: err     - cross-validation test error
%      err_std - standard error of the mean

% bb 06/03, ida.first.fhg.de


cnt= proc_selectChannels(cnt, method.chans);
epo= makeEpochs(cnt, mrk, [-method.ilen 0] + tcl, method.jit);
if ~method.separateActionClasses,
  epo.y= ones(1, size(epo.y,2));
  epo.className= {'action'};
end
no_moto= makeEpochs(cnt, mrk, [-method.ilen 0], method.jit_noevent);
no_moto.y= ones(1,size(no_moto.y,2));
no_moto.className= {'no event'};

epo= proc_appendEpochs(epo, no_moto);
eval(method.proc);

%% only loss for confusing "action" with "no action"
nClasses= size(epo.y,1);
fv.loss= zeros(nClasses, nClasses);
fv.loss(:,nClasses)= 1;
fv.loss(nClasses,:)= 1;
fv.loss(nClasses,nClasses)= 0;

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
