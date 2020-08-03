function [outarg1, errStd, outTe, avErr, evErrTe, evErrTr]= ...
    doXvalidationPlus(epo, classy, xTrials, verbose)
%[errMean, errStd, outTe, avErr]= ...
%                  doXvalidationPlus(fv, model, xTrials, <verbose>)
%
% IN   fv      - features vectors, struct with fields
%                .x (data, nd array where last dim is nSamples) and
%                .y (labels [nClasses x nSamples], 1 means membership)
%                ('substructure' of data structure of epoched data)
%      model   - name or structure specifying a classification model
%                cd to bci/classifiers and type 'help Contents'
%                if the model has free parameters, they are chosen by
%                selectModel on each training set (time consuming!)
%      xTrials - [#trials, #splits]
%                if a third value #samples is specified, for each trial
%                a subset containing #samples is drawn from fv
%      verbose - level of verbosity
%
% some additional features can be evoked by adding fields to the fv structure:
%
%    .proc     - class dependent preprocessing; is executed for each
%                training set (labels of the test set are set to zero),
%                can be given as code or file name of a tape
%                the procedure must calculate 'fv' from 'epo'
%    .test_jits- use only samples corresponding to those jitters for testing,
%                cf. bci/demos/jittered_training
%    .loss     - calculate error due to loss matrix [Classes x nClasses]
%    .fp_bound - fix FP-rate on test set (separating hyperplane classys only)
%    .equi     - subsets that should be equilibrated
%    .testset  - a index set where test sets can only come from
%                (see sampleDivisions)
%
% OUT  errMean - [testError, trainingError]
%      errStd  - analog, standard error OF THE MEANS
%      outTe   - continuous classifier output for each x-val trial / sample
%      avErr   - average error in each x-val trial
%
% GLOBZ  NO_CLOCK
%
% SEE selectModel, sampleDivisions

% bb, ida.first.fhg.de
% with extensions by guido

global NO_CLOCK

if exist('verbose', 'var') & ischar(verbose) & ...
      strcmpi(verbose, 'train only')==1,
  TRAIN_ONLY= 1; 
  if verbose(1)=='T',
    verbose= 2;
  else
    verbose= 0;
  end
else
  TRAIN_ONLY= 0;
end
if ~exist('verbose', 'var'), verbose= 0; end

if length(xTrials)>=3 & xTrials(3)<0,
  xTrials(3)= round((xTrials(2)+xTrials(3))/xTrials(2)*sum(any(epo.y)));
end


[nClasses, nEvents]= size(epo.y);
if isfield(epo, 'loss'),
  loss= epo.loss;
else
  loss= ones(nClasses,nClasses) - eye(nClasses);
end

if isfield(epo, 'bidx'),
  baseIdx= unique(epo.bidx);
else
  baseIdx= 1:size(epo.y,2);
  epo.bidx= baseIdx;
end
if ~isfield(epo, 'jit'),
  epo.jit= zeros(size(epo.bidx));
end
if ~isfield(epo, 'train_jits'),
  epo.train_jits= unique(epo.jit);
end
if ~isfield(epo, 'test_jits'),
  epo.test_jits= unique(epo.jit);
end

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
  epo= rmfield(epo, 'divTr');  %% otherwise it make trouble in model selection
  epo= rmfield(epo, 'divTe');
else
  if isfield(epo,'testset')
    testset = epo.testset;
  else
    testset = [];
  end
  if isfield(epo, 'equi'),
    [divTr, divTe]= sampleDivisions(epo.y(:,baseIdx), xTrials, epo.equi,testset);
  else
    [divTr, divTe]= sampleDivisions(epo.y(:,baseIdx), xTrials,[],testset);
  end
end

nTrials= xTrials(1);
nDivisions= xTrials(2);
if ~TRAIN_ONLY,
  avErr= zeros(nTrials,length(divTe{1}), 2);
  evErrTe= zeros(1, nEvents);
  evErrTr= zeros(1, nEvents);
  nTe= zeros(1, nEvents);
  nTr= zeros(1, nEvents);
  outTe= zeros(nTrials, nEvents);
end
y_memo= epo.y;
t0= cputime;  %tic;
for n= 1:nTrials,
  nDiv= length(divTe{n});  %% might differ from nDivisons in 'loo' case
  for d= 1:nDiv,
    k= d+(n-1)*nDiv;
    bidxTr= divTr{n}{d};
    bidxTe= divTe{n}{d};
    idxTr= find(ismember(epo.bidx, bidxTr) & ...
                ismember(epo.jit, epo.train_jits));
    idxTe= find(ismember(epo.bidx, bidxTe) & ...
                ismember(epo.jit, epo.test_jits));
    epo.y(:,idxTe)= 0;                       %% hide labels of test set
    if ~isempty(model),       %% do model selection on training set
      noc = NO_CLOCK;
      NO_CLOCK = 1;
      [classy, E, V]= selectModel(epo, model, [], max(0,verbose-1));
      NO_CLOCK = noc;
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
    daten= setTrainset(fv, idxTr);
    C= feval(trainFcn, daten.x, daten.y, params{:});
    
    if fp_bound,
      idxTeNeg=  idxTe(find(epo.y(1,idxTe)));
      frac= floor(length(idxTeNeg)*fp_bound);
      xp= C.w'*fv.x(:,idxTeNeg);
      [so,si]= sort(-xp);
      C.b= so(frac+1) - eps;
    end
    if TRAIN_ONLY,
      outarg1(k)= C;
    else
      out= feval(applyFcn, C, fv.x);  
      outTe(n, idxTe)= out(idxTe);
      if size(out,1)==1,
        if nClasses==2,  %% & any(out<0),
          out= 1.5 + 0.5*sign(out);
        end
      else
        [dummy, out]= max(out);
      end
      if nClasses>2,
        outTe(n, idxTe)= out(idxTe);  %% TODO: return detailed output
      end
      errCl= loss(sub2ind(size(loss), label, out));
      avErr(n,d,:)= [mean(errCl(idxTe)) mean(errCl(idxTr))];

      evErrTe(idxTe)= evErrTe(idxTe) + errCl(idxTe);
      evErrTr(idxTr)= evErrTr(idxTr) + errCl(idxTr);
      nTe(idxTe)= nTe(idxTe) + 1;
      nTr(idxTr)= nTr(idxTr) + 1;
    end
    if NO_CLOCK, 
%      print_progress(k, nDiv*nTrials);
    else 
      showClock(k, nDiv*nTrials); 
    end
  end  
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

avE = mean(avErr,2);
avErr = reshape(avErr,[nTrials*length(divTe{1}),2]);
errMean= mean(100*avErr, 1);
errStd= transpose(squeeze(std(100*avE, 0, 1)));
outarg1= errMean;

if nargout==0 | verbose,
  if verbose>2,
    [mc, me, ms]= calcConfusionMatrix(epo, outTe);
    fprintf(['%4.1f' 177 '%.1f%%  (fn: %4.1f' 177 '%.1f%%,  fp: %4.1f' 177 ...
             '%.1f%%)  [train: %.1f' 177 '%.1f%%]' ...
             '  (%.1f s for [%s] trials)\n'], ...
            errMean(1), errStd(1), me(2), ms(2), me(3), ms(3), ...
            errMean(2), errStd(2), et, vec2str(round(xTrials),'%d',' '));
  else
    fprintf(['%4.1f' 177 '%.1f%%, [train: %4.1f' 177 '%.1f%%]' ...
             '  (%.1f s for [%s] trials)\n'], ...
              [errMean; errStd], et, vec2str(round(xTrials),'%d',' '));
  end
end
