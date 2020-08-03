function [dat, varargout]= proc_csp_auto(dat, varargin)
%PROC_CSP_AUTO - Common Spatial Pattern Analysis with Auto Filter Selection
%
%Description:
% calculate common spatial patterns (CSP).
% please note that this preprocessing uses label information. so for
% estimating a generalization error based on this processing, you must
% not apply csp to your whole data set and then do cross-validation
% (csp would have used labels of samples in future test sets).
% you should use the .proc (train/apply) feature in xvalidation, see
% demos/demo_validation_csp
%
%Synopsis:
% [DAT, CSP_W, CSP_EIG, CSP_A]= proc_csp_auto(DAT, <OPT>);
% [DAT, CSP_W, CSP_EIG, CSP_A]= proc_csp_auto(DAT, NPATS);
%
%Arguments:
% DAT    - data structure of epoched data
% NPATS  - number of patterns per class to be calculated, 
%          deafult nChans/nClasses.
% OPT - struct or property/value list of optional properties:
%  .patterns - 'all' or matrix of given filters or number of filters. 
%      Selection depends on opt.selectPolicy.
%  .selectPolicy - determines how the patterns are selected. Valid choices are 
%      'all', 'equalperclass', 
%      'maxvalues' (attention: possibly not every class is represented!)
%      'maxvalueswithcut' (like maxvalues, but components with less than
%      half the maximal score are omitted),
%      'directorscut' (default, heuristic, similar to maxvalueswithcut, 
%      but every class will be represented), 
%      'matchfilters' (in that case opt.patterns must be a matrix),
%      'matchpatterns' (not implemented yet)
%  .covPolicy - 'normal' or 'average' (default). The latter calculates
%       the average of the single-trial covariance matrices.
%  .score - 'eigenvalues', 'medianvar' (default), 'roc', 'fisher' to
%       determine the components by the respective score
%
%Returns:
% DAT    - updated data structure
% CSP_W  - CSP projection matrix (filters)
% CSP_EIG- eigenvalue score of CSP projections (rows in W)
% CSP_A  - estimated mixing matrix (activation patterns)
%
%See also demos/demo_validate_csp

% Author(s): Benjamin Blankertz
 
[T, nChans, nEpochs]= size(dat.x);

if size(dat.y,1)~=2,
  error('this function works only for 2-class data.');
end

if length(varargin)==1 & isnumeric(varargin{1}),
  opt= struct('patterns', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'patterns', 3, ...
                  'score', 'medianvar', ...
                  'covPolicy', 'average', ...
                  'scaling', 'none', ...
                  'normalize', 0, ...
                  'selectPolicy', 'directorscut',...
                  'weight',ones(1,size(dat.y,2)),...
                  'weight_exp',1);

%% calculate classwise covariance matrices
if isnumeric(opt.covPolicy),
  R= opt.covPolicy,
  if ~isequal(size(R), [nChans nChans 2]),
    error('precalculated covariance matrices have wrong size');
  end
else
  R= procutil_covClasswise(dat, opt);
end

%% do actual CSP calculation as generalized eigenvalues
[W,D]= eig(R(:,:,2),R(:,:,1)+R(:,:,2));

%% calculate score for each CSP channel
switch(lower(opt.score)),
 case 'eigenvalues',
  score= diag(D);
 case 'medianvar',
  fv= proc_linearDerivation(dat, W);
  fv= proc_variance(fv);
  score= zeros(nChans, 1);
  c1= find(fv.y(1,:));
  c2= find(fv.y(2,:));
  for kk= 1:nChans,
    v1= median(fv.x(1,kk,c1),3);
    v2= median(fv.x(1,kk,c2),3);
    score(kk)= v2/(v1+v2);
  end
 case 'roc',
  fv= proc_linearDerivation(dat, W);
  fv= proc_variance(fv);
  fv= proc_rocAreaValues(fv);
%  score= abs(fv.x);
  score= -fv.x;
 case 'fisher',
  fv= proc_linearDerivation(dat, W);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  fv= proc_rfisherScore(fv, 'preserve_sign',1);
%  score= abs(fv.x);
  score= -fv.x;
 otherwise,
  error('unknown option for score');
end
%% force score to be a column vector
score= score(:);

%% select patterns
if ischar(opt.patterns) & strcmpi(opt.patterns, 'all'),
  fi= 1:nChans;
elseif ischar(opt.patterns) & strcmpi(opt.patterns, 'auto'),
  if ~strcmpi(opt.selectPolicy, 'maxvalues'),
    score= max(score, 1-score);
  end
  perc= percentiles(score, [20 80]);
  thresh= perc(2) + diff(perc);
  fi= find(score>thresh);
else
  switch(lower(opt.selectPolicy)),
   case 'all',
    fi= 1:nChans;
   case 'equalperclass',
    [dd,di]= sort(score);
    fi= [di(1:opt.patterns); di(end:-1:nChans-opt.patterns+1)];
   case 'minclass2',
    [dd,di]= sort(score);
    fi= di(1:opt.patterns);
   case 'minclass1',
    [dd,di]= sort(score);
    fi= di(end:-1:nChans-opt.patterns+1);
   case 'bestvalues',
    [dd,di]= sort(min(score, 1-score));
    fi= di(1:opt.patterns);
   case 'maxvalues',
    [dd,di]= sort(-score);
    fi= di(1:opt.patterns);
   case 'maxvalueswithcut',
    score= score/max(score);
    [dd,di]= sort(-score);
    iMax= 1:opt.patterns;
    iCut= find(-dd>=0.5);
    idx= intersect(iMax, iCut);
    fi= di(idx);
   case 'directorscut',
    if ismember(opt.score, {'eigenvalues','medianvar'}),
      absscore= 2*(max(score, 1-score)-0.5);
      [dd,di]= sort(score);
      Nh= floor(nChans/2);
      iC1= find(ismember(di, 1:Nh));
      iC2= flipud(find(ismember(di, [nChans-Nh+1:nChans])));
      iCut= find(absscore(di)>=0.66*max(absscore));
      idx1= [iC1(1); intersect(iC1(2:opt.patterns), iCut)];
      idx2= [iC2(1); intersect(iC2(2:opt.patterns), iCut)];
      fi= di([idx1; flipud(idx2)]);
    else
      score= score/max(score);
      [dd,di]= sort(-score);
      Nh= floor(nChans/2);
      iC1= find(ismember(di, 1:Nh));
      iC2= find(ismember(di, [nChans-Nh+1:nChans]));
      iCut= find(-dd>=0.5);
      idx1= [iC1(1); intersect(iC1(2:opt.patterns), iCut)];      
      idx2= [iC2(1); intersect(iC2(2:opt.patterns), iCut)];
      fi= di([idx1; idx2]);
    end
   case 'matchpatterns',  %% to be implemented
    error('to be implemented');
   case 'matchfilters',  %% greedy, not well implemented
    fi= zeros(1,size(opt.patterns,2));
    for ii= 1:size(opt.patterns,2),
      v1= opt.patterns(:,ii);
      v1= v1/sqrt(v1'*v1);
      sp= -inf*ones(1,nChans);
      for jj= 1:nChans,
        if ismember(jj, fi), continue; end
        v2= W(:,jj);
        sp(jj)= abs(v1'*v2/sqrt(v2'*v2));
      end
      [mm,mi]= max(sp);
      fi(ii)= mi;
    end
   otherwise,
    error('unknown selectPolicy');
  end
end

Wp= W(:,fi);
la= score(fi);

%% optional scaling of CSP filters to make solution unique
switch(lower(opt.scaling)),
 case 'maxto1',
  for kk= 1:size(Wp,2),
    [mm,mi]= max(abs(Wp(:,kk)));
    Wp(:,kk)= Wp(:,kk) / Wp(mi,kk);
  end
 case 'none',
 otherwise
  error('unknown scaling');
end


%% save old channel labels
if isfield(dat, 'clab'),
  dat.origClab= dat.clab;
end

%% apply CSP filters to time series
dat= proc_linearDerivation(dat, Wp, 'prependix','csp');

%% arrange optional output arguments
if nargout>1,
  varargout{1}= Wp;
  if nargout>2,
    varargout{2}= la;
    if nargout>3,
      A= pinv(W);
      varargout{3}= A(fi,:);
      if nargout>4,
        varargout{4}= fi;
      end
    end
  end
end
