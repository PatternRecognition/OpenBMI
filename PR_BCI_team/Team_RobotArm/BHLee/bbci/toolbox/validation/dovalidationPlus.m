function [outarg1, errStd, outTe, avErr, evErrTe, evErrTr]= ...
    dovalidationPlus(epo, classy, xTrials, verbose)
%[errMean, errStd, outTe]= doXvalidationPlus(epo, classy, xTrials, <verbose>)
%
% some additional features can be evoked by adding fields to the epo structure:
%
%    .loss     - calculate error due to loss matrix (nClasses x nClasses)
%    .fp_bound - fix FP-rate on test set (separating hyperplane classys only)
%    .nJits    - number of (jittered) versions of each event, incl. original
%    .equi     - subsets that should be equilibrated
%    .proc     - class dependent preprocessing; is executed for each
%                training set (labels of the test set are set to zero),
%                can be given as code or file name of a tape
%                the procedure must calculate 'fv' from 'epo'
%
% GLOBZ  NO_CLOCK

global NO_CLOCK

if exist('verbose', 'var') & ischar(verbose) & ...
      strmatch(verbose, 'train only')==1,
  TRAIN_ONLY= 1;
  verbose= 0;
else
  TRAIN_ONLY= 0;
end
if ~exist('verbose', 'var'), verbose= 0; end

[nClasses, nEvents]= size(epo.y);
if isfield(epo, 'loss'),
  loss= epo.loss;
else
  loss= ones(nClasses,nClasses) - eye(nClasses);
end
if isfield(epo, 'nJits'),
  nJits= epo.nJits;
else
  nJits= 1;
end
nBase= round(nEvents/nJits);
if isfield(epo, 'proc') & ~isempty(epo.proc),
  if exist(epo.proc, 'file'),
    if isfield(epo, 'proc_no'),
      proc_no= epo.proc_no;
    else
      proc_no= 1;
    end
    proc= getBlockFromTape(epo.proc, proc_no);
  else
    proc= epo.proc;
  end
  proc= [proc ', fv= proc_flaten(fv);'];
else
  proc= 'fv= proc_flaten(epo);';
end

if isstruct(classy),       %% classification model with free hyper parameters
  model= classy;
  classy= model.classy;
  model_idx= getModelParameterIndex(model.classy);
else
  model= [];
end

[func, params]= getFuncParam(classy);
trainFcn= ['train_' func];
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end

if isfield(epo, 'fp_bound'),
  if ~isequal(applyFcn, 'apply_separatingHyperplane'),
    error('FP-bound works only for separating hyperplane classifiers');
  end
  fp_bound= epo.fp_bound;
else
  fp_bound= 0;
end

[dummy, label]= max(epo.y);
if isfield(epo, 'divTe'),
  divTr= epo.divTr;
  divTe= epo.divTe;
  xTrials= [length(divTe) length(divTe{1})];
else
  if isfield(epo, 'equi'),
    [divTr, divTe]= sampDivisions(epo.y(:,1:nBase), xTrials, epo.equi);
  else
    [divTr, divTe]= sampDivisions(epo.y(:,1:nBase), xTrials);
  end
end
nTrials= xTrials(1);
nTrain= xTrials(2);
if ~TRAIN_ONLY,
  avErr= zeros(nTrials, 2);
  evErrTe= zeros(1, nEvents);
  evErrTr= zeros(1, nEvents);
  nTe= zeros(1, nEvents);
  nTr= zeros(1, nEvents);
  outTe= zeros(nTrials, nEvents);
end
y_memo= epo.y;
t0= cputime;
for n= 1:nTrials,
  bidxTr= divTr{n};
  bidxTe= divTe{n};
  idxTr= jitteredIndices(bidxTr, (1:nJits-1)*nBase);
  idxTe= jitteredIndices(bidxTe, (1:nJits-1)*nBase);
  epo.y(:,idxTe)= 0;                       %% hide labels of test set
  if ~isempty(model),       %% do model selection on training set
    [classy, E, V]= selectModel(epo, model, [], max(0,verbose-1));
    [func, params]= getFuncParam(classy);
    if verbose>1,
      fprintf(['selection chose: %g -> %.1f' 177 '%.1f%%\n'], ...
	      classy{model_idx}, E, V(1));
    end
  end
  
  eval(proc);                              %% epo -> fv
  
  epo.y= y_memo;
  % the following update is for feature combination.
  % DUDU, 03.07.02
  daten = setTrainset(fv,idxTr);
  C= feval(trainFcn, daten.x, daten.y, params{:});
    
  if fp_bound,
    idxTeNeg=  idxTe(find(epo.y(1,idxTe)));
    frac= floor(length(idxTeNeg)*fp_bound);
    xp= C.w'*fv.x(:,idxTeNeg);
    [so,si]= sort(-xp);
    C.b= so(frac+1) - eps;
    end
    if TRAIN_ONLY,
      outarg1(n)= C;
    else
      out= feval(applyFcn, C, fv.x);  
      outTe(n, idxTe)= out(idxTe);
      if size(out,1)==1, 
        out= 1.5 + 0.5*sign(out);
      else
        [dummy, out]= max(out);
      end
      errCl= loss(sub2ind(size(loss), label, out));
      avErr(n, :)= [mean(errCl(bidxTe)) mean(errCl(bidxTr))];
      
      evErrTe(idxTe)= evErrTe(idxTe) + errCl(idxTe);
      evErrTr(idxTr)= evErrTr(idxTr) + errCl(idxTr);
      nTe(idxTe)= nTe(idxTe) + 1;
      nTr(idxTr)= nTr(idxTr) + 1;
    end
    if NO_CLOCK, else showClock(n, nTrials); end
end  

if TRAIN_ONLY,
  return;
end

et= cputime-t0;
nTe(find(nTe==0))= 1;
nTr(find(nTr==0))= 1;
evErrTe= evErrTe./nTe;
evErrTr= evErrTr./nTr;

if length(xTrials)==3 & xTrials(3)~=0,  %% trials are chosen unequally
  evErrTe= [];
  evErrTr= [];
end

errMean= mean(100*avErr, 1);
errStd= std(100*avErr, 0, 1);
outarg1= errMean;

if nargout==0 | verbose,
  fprintf(['%4.1f' 177 '%.1f%%, [train: %4.1f' 177 '%.1f%%]' ...
           '  (%.1f s for [%s] trials)\n'], ...
          [errMean; errStd], et, vec2str(round(xTrials)));
end

