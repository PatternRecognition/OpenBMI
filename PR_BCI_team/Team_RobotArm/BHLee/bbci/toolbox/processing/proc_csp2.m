function [dat, Wp, la, A]= proc_csp2(dat, varargin)
%PROC_CSP2 - Common Spatial Pattern Analysis for 2 classes
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
% [DAT, W, LA, A]= proc_csp(DAT, <OPT>);
% [DAT, W, LA, A]= proc_csp(DAT, NPATS);
%
%Input:
% DAT    - data structure of epoched data
% NPATS  - number of patterns per class to be calculated, 
%          deafult nChans/nClasses.
% OPT - struct or property/value list of optional properties:
%  .patternsPerClass
%  .covPolicy - 'normal' or 'average' (default). The latter calculates
%          the average of the single-trial covariance matrices.
%
%Output:
% DAT    - updated data structure
% W      - CSP projection matrix (filters)
% LA     - eigenvalue score of CSP projections (rows in W)
% A      - estimated mixing matrix (activation patterns)
%
%See also demos/demo_validate_csp

% Author(s): Benjamin Blankertz, long time ago
 
[T, nChans, nEpochs]= size(dat.x);
nClasses= size(dat.y,1);

if nClasses~=2,
  error('this function works only for 2-class data.');
end

if length(varargin)==1 & isnumeric(varargin{1}),
  opt= struct('patterns', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'patterns', 'all', ...
                  'covPolicy', 'average', ...
                  'selectPolicy', 'equalPerClass');

R = zeros(nChans, nChans, nClasses);
switch(lower(opt.covPolicy)),
 case 'average',
  for t= 1:nClasses,
    C= zeros(nChans, nChans);
    idx= find(dat.y(t,:)>0);
    for m= idx,
      C= C + cov(dat.x(:,:,m));
    end
    R(:,:,t)= C/length(idx);
  end
 case 'normal',
  for t= 1:nClasses,
    idx= find(dat.y(t,:)>0);
    x= permute(dat.x(:,:,idx), [1 3 2]);
    x= reshape(x, T*length(idx), nChans);
    R(:,:,t) = cov(x);
  end
 otherwise,
  error('covPolicy not known');
end

[U,D]= eig(sum(R,3)); 
P= diag(1./sqrt(diag(D)))*U';
[B,D]= eig(P*R(:,:,2)*P');
W= P'*B;
diagD= diag(D);
if strcmpi(opt.patterns, 'all'),
  fi= 1:nChans;
else
  switch(lower(opt.selectPolicy)),
   case 'all',
    fi= 1:nChans;
   case 'equalperclass',
    [dd,di]= sort(diagD);
    fi= [di(1:opt.patterns); di(end:-1:nChans-opt.patterns+1)];
   case 'besteigenvalues',
    [dd,di]= sort(min(diagD, 1-diagD));
    fi= di(1:opt.patterns);
   otherwise,
    error('unknown selectPolicy');
  end
end
Wp= W(:,fi);
la= max(diagD(fi), 1-diagD(fi));

nNewChans= size(Wp, 2);
xx= zeros(T, nNewChans, nEpochs);
for m= 1:nEpochs,
  xx(:,:,m)= dat.x(:,:,m)*Wp;
end
dat.x= xx;

dat.origClab= dat.clab;
dat.clab= cellstr([repmat('csp',nNewChans,1) num2str((1:nNewChans)')])';
%dat.clab= cell(1, nNewChans);
%if ~isfield(dat, 'className'),
%  dat.className= cellstr([repmat('cl',nClasses,1) num2str((1:nClasses)')]);
%end
%cCount= [0 0];
%for ii= 1:nNewChans,
%  cc= 1 + (fi(ii)>nChans/2);
%  cCount(cc)= cCount(cc) + 1;
%  dat.clab{ii}= sprintf('%s:csp%d', dat.className{cc}, cCount(cc));
%end

if nargout>3,
  A= pinv(W);
  A= A(fi,:);
end
if nargout<3,
  clear la;
  if nargout<2,
    clear Wp;
  end
end
