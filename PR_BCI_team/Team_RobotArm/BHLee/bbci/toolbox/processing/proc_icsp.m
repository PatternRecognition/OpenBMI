function [dat, varargout]= proc_icsp(dat, disturb, varargin)
%PROC_ICSP - Invariant Common Spatial Pattern Analysis
%
%Description:
% calculate invariant common spatial patterns (iCSP).
% please note that this preprocessing uses label information. so for
% estimating a generalization error based on this processing, you must
% not apply csp to your whole data set and then do cross-validation
% (csp would have used labels of samples in future test sets).
% you should use the .proc (train/apply) feature in xvalidation, see
% demos/demo_validation_csp
%
%Synopsis:
% [DAT, CSP_W, CSP_EIG, CSP_A, DIS_EIG]= proc_icsp(DAT, DAT_DISTURB, <OPT>);
% [DAT, CSP_W, CSP_EIG, CSP_A, DIS_EIG]= proc_icsp(DAT, DAT_DISTURB, NPATS);
%
%Arguments:
% DAT    - data structure of epoched data
% DAT_DISTRUB - data structure of epoched data whose impact on the
%          CSP filtered channels should be minimized
% NPATS  - number of patterns per class to be calculated, 
%          deafult nChans/nClasses.
% OPT - struct or property/value list of optional properties:
%  .nu  - factor for disturbance minimization, default 0.5
%  .patterns - 'all' or matrix of given filters or number of filters. 
%      Selection depends on opt.selectPolicy.
%  .selectPolicy - determines how the patterns are selected. Valid choice
%      are 'equalperclass' (default), 'besteigenvalues', 'all',
%      'matchfilters' (in that case opt.patterns must be a matrix)
%      'matchpatterns' (not implemented yet)
%
%Returns:
% DAT    - updated data structure
% CSP_W  - CSP projection matrix (filters)
% CSP_EIG- eigenvalue score of CSP filters (columns in W)
% CSP_A  - estimated mixing matrix (activation patterns)
% DIS_EIG- eigenvalues of CSP filters with respect to disturbance covariance
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
                  'nu', 0.5, ...
                  'patterns', 'all', ...
                  'scaling', 'none', ...
                  'selectPolicy', 'equalPerClass', ...
                  'normalize_by_trace', 0);

%% calculate classwise covariance matrices
R = zeros(nChans, nChans, 2);
for t= 1:2,
  idx= find(dat.y(t,:));
  x= permute(dat.x(:,:,idx), [1 3 2]);
  x= reshape(x, T*length(idx), nChans);
  R(:,:,t) = cov(x);
end
if opt.normalize_by_trace,
  tr= mean([trace(R(:,:,1)), trace(R(:,:,2))]);
  R= R/tr;
end

sx= size(disturb.x);
x= permute(disturb.x, [1 3 2]);
x= reshape(x, [sx(1)*sx(3) sx(2)]);
Psi= cov(x);
if opt.normalize_by_trace,
  Psi= Psi / trace(Psi);
end

%% regularize if neccessary
if opt.nu==1 & rank(Psi)<nChans,
  Psi= Psi + eye(nChans,nChans)*trace(Psi)/nChans/1e5;
end

%% do actual iCSP calculation as generalized eigenvalues
[W1,D1]= eig(R(:,:,2), (1-opt.nu)*(R(:,:,1)+R(:,:,2)) + opt.nu*Psi);
W1= fliplr(W1);
D1= diag(flipud(diag(D1)));
[W2,D2]= eig(R(:,:,1), (1-opt.nu)*(R(:,:,1)+R(:,:,2)) + opt.nu*Psi);
W= [W1,W2];
diagD= [diag(D1); diag(D2)];

if any(isinf(diagD)),
  warning('some eigs are inf');
  idx= find(isinf(diagD));
  diagD(idx)= 0;
  [so,si]= sort(-diagD(1:nChans));
  diagD(1:nChans)= diagD(si);
  W(:,1:nChans)= W(:,si);
  [so,si]= sort(diagD(nChans+1:end));
  diagD(nChans+1:end)= diagD(si+nChans);
  W(:,nChans+1:end)= W(:,si+nChans);
end


%% select patterns
if strcmpi(opt.patterns, 'all'),
  fi= [1:opt.patterns, 2*nChans+1-[1:floor(nChans/2)]];
else
  switch(lower(opt.selectPolicy)),
   case 'all',
    fi= [1:opt.patterns, 2*nChans+1-[1:floor(nChans/2)]];
   case 'equalperclass',
    fi= [1:opt.patterns, 2*nChans+1-[1:opt.patterns]];
   case 'besteigenvalues',
    [dd,di]= sort(-diagD);
    fi= di(1:opt.patterns);
   case 'matchpatterns',  %% greedy, not well implemented
    fi= zeros(1,size(opt.patterns,2));
    for ii= 1:size(opt.patterns,2),
      v1= opt.patterns(:,ii);
      v1= v1/sqrt(v1'*v1);
      sp= -inf*ones(1,nChans);
      for jj= 1:2*nChans,
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
la= diagD(fi);

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
      A= [pinv(W1); pinv(W2)];
      varargout{3}= A(fi,:);
      if nargout>4,
        ed1= diag(W1'*Psi*W1);
        ed2= diag(W2'*Psi*W2);
        eig_distrub= [ed1; ed2];
        varargin{4}= eig_distrub(fi);
      end
    end
  end
end
