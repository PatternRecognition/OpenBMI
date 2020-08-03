function [dat, varargout]= proc_rcsp(dat, varargin)
%PROC_CSP3 - Common Spatial Pattern Analysis for 2 classes
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
% [DAT, CSP_W, CSP_EIG, CSP_A]= proc_csp3(DAT, <OPT>);
% [DAT, CSP_W, CSP_EIG, CSP_A]= proc_csp3(DAT, NPATS);
%
%Arguments:
% DAT    - data structure of epoched data
% NPATS  - number of patterns per class to be calculated, 
%          deafult nChans/nClasses.
% OPT - struct or property/value list of optional properties:
%  .patterns - 'all' or matrix of given filters or number of filters. 
%      Selection depends on opt.selectPolicy.
%  .selectPolicy - determines how the patterns are selected. Valid choice
%      are 'equalperclass' (default), 'besteigenvalues', 'all',
%      'matchfilters' (in that case opt.patterns must be a matrix)
%      'matchpatterns' (not implemented yet)
%  .covPolicy - 'normal' or 'average' (default). The latter calculates
%          the average of the single-trial covariance matrices.
%  .weight    - vector of length size(dat.y,2). When averaging training 
%               trial covariances, this weight is multiplied with the
%               corresponding trial. (default ones)
%  .weight_exp - exponent for weight. (default 1);
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
                  'patterns', 'all', ...
                  'score', 'eigenvalues', ...
                  'covPolicy', 'average', ...
                  'scaling', 'none', ...
                  'normalize', 0, ...
                  'selectPolicy', 'equalPerClass',...
                  'weight',ones(1,size(dat.y,2)),...
                  'weight_exp',1);

if strmatch(opt.score, {'roc','fisherscore'}),
  if isempty(strpatternmatch({'maxvalues*','directorscut'}, opt.selectPolicy)),
    warning('opt.selectPolicy forced to ''maxvalues''.');
    opt.selectPolicy= 'maxvalues';
  end
end

%% calculate classwise covariance matrices
R = zeros(nChans, nChans, 2);
switch(lower(opt.covPolicy)),
 case 'average',
  for t= 1:2,
    C= zeros(nChans, nChans);
    idx= find(dat.y(t,:));
    for m= idx,
      C= C + (opt.weight(m)^opt.weight_exp)*cov(dat.x(:,:,m));
    end
    R(:,:,t)= C/length(idx);
  end
 case 'normal',
  for t= 1:2,
    idx= find(dat.y(t,:));
    x= permute(dat.x(:,:,idx), [1 3 2]);
    x= reshape(x, T*length(idx), nChans);
    R(:,:,t) = cov(x);
  end
 otherwise,
  error('covPolicy not known');
end

if opt.normalize,
  t1= trace(R(:,:,1));
  t2= trace(R(:,:,2));
  if opt.normalize==1,
    R(:,:,1)= R(:,:,1)/t1;
    R(:,:,2)= R(:,:,2)/t2;
  elseif opt.normalize==2,
    R(:,:,1)= R(:,:,1)/mean([t1 t2]);
    R(:,:,2)= R(:,:,2)/mean([t1 t2]);
  else
    error('idiot');
  end
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
  score= abs(fv.x);
  keyboard
 case 'fisher',
  fv= proc_linearDerivation(dat, W);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  fv= proc_rfisherScore(fv, 'preserve_sign',1);
%  score= fv.x;
  score= abs(fv.x);
 otherwise,
  error('unknown option for score');
end

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
    score= score/max(score);
    [dd,di]= sort(-score);
    Nh= floor(nChans/2);
    iC1= find(ismember(di, 1:Nh));
    iC2= find(ismember(di, [nChans-Nh+1:nChans]));
    iCut= find(-dd>=0.5);
    idx1= [iC1(1), intersect(iC1(2:opt.patterns), iCut)];
    idx2= [iC2(1), intersect(iC2(2:opt.patterns), iCut)];
    fi= di([idx1 idx2]);
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
