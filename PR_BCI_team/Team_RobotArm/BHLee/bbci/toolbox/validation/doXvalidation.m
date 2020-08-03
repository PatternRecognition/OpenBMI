function [errMean, errStd, outTe, avErr]= ...
    doXvalidation(dat, classy, xTrials, verbose)
%[errMean, errStd, outTe]= doXvalidation(fv, classy, xTrials, <verbose>)
%
% calculate the cross-validation error of classifier 'classy' on
% the set of feature vectors 'fv'. see doXvalidationPlus which
% offers a lot more functions.
%
% IN   fv      - features vectors, struct with fields
%                .x (data, nd array where last dim is nSamples) and
%                .y (labels [nClasses x nSamples], 1 means membership)
%                ('substructure' of data structure of epoched data)
%      classy  - name or structure specifying a classification model
%                (must not contain free model parameters, see xValidationPlus)
%                cd to bci/classifiers and type 'help Contents'
%      xTrials - [#trials, #splits]
%                if a third value #samples is specified, for each trial
%                a subset containing #samples is drawn from fv
%      verbose - level of verbosity
%
% OUT  errMean - [testError, trainingError]
%      errStd  - analog, standard error OF THE MEANS
%      outTe   - continuous classifier output for each x-val trial
%
% GLOBZ  NO_CLOCK
%
% SEE doXvalidationPlus, sampleDivisions

% bb, ida.first.fhg.de


global NO_CLOCK

dat= proc_flaten(dat);
[func, params]= getFuncParam(classy);
trainFcn= ['train_' func];
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end

nEvents= size(dat.y, 2);
[dummy, label]= max(dat.y);
[divTr, divTe]= sampleDivisions(dat.y, xTrials);
nTrials= xTrials(1);
nDivisions= xTrials(2);
avErr= zeros(nTrials, length(divTe{1}), 2);
outTe= zeros(nTrials, nEvents);

t0= cputime; %tic;
for n= 1:nTrials,
  nDiv= length(divTe{n});  %% might differ from nDivisons in 'loo' case
  for d= 1:nDiv,
    k= d+(n-1)*nDiv;
    idxTr= divTr{n}{d};
    idxTe= divTe{n}{d};
    C= feval(trainFcn, dat.x(:,idxTr), dat.y(:,idxTr), params{:});
    out= feval(applyFcn, C, dat.x);
    if size(out,1)==1, 
      out= 1.5 + 0.5*sign(out);
    else
      [dummy, out]= max(out);
    end
    outTe(n, idxTe)= out(idxTe);
    errCl= (out~=label);
    avErr(n, d, :)= [mean(errCl(idxTe)) mean(errCl(idxTr))];
    if NO_CLOCK, 
%      print_progress(k, nDiv*nTrials);
    else 
      showClock(k, nDiv*nTrials); 
    end
  end  
end
et= cputime-t0;

%errMean= mean(100*avErr);
%errStd= std(100*avErr);
avE = mean(avErr,2);
avErr = reshape(avErr,[nTrials*length(divTe{1}),2]);
errMean= mean(100*avErr, 1);
errStd= transpose(squeeze(std(100*avE, 0, 1)));

if nargout==0 | (exist('verbose','var') & verbose),
  fprintf(['%4.1f' 177 '%.1f%%, [train: %4.1f' 177 '%.1f%%]' ...
           '  (%.1f s for [%s] trials)\n'], ...
          [errMean; errStd], et, vec2str(round(xTrials),'%d',' '));
end
