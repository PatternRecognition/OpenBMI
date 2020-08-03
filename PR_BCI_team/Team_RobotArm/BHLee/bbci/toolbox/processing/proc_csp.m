function [dat, Wp, la]= proc_csp(dat, nComps, COV_AVG, NORMALIZE)
%[dat, W, la]= proc_csp(dat, <nComps=nChans/nClasses, COV_AVG=1, NORMALIZE=0>);
%
% calculate common spatial patterns (CSP).
% please note that this preprocessing uses label information. so for
% estimating a generalization error based on this processing, you must
% not apply csp to your whole data set and then do cross-validation
% (csp would have used labels of samples in future test sets).
% you should use the .proc feature in doXvalidationPlus, see
% demos/classification_csp
%
% IN   epo       - data structure of epoched data
%      nComps    - number of patterns to be calculated, deafult nChans/nClasses
%      COV_AVG   - average covariance matrices? default 1
%      NORMALIZE - normalize covariance matrices? default 0
%
% OUT  epo       - updated data structure
%      W         - CSP projection matrix
%      la        - eigenvalue score of CSP projections (rows in W)
%
% SEE demos/classification_csp

% bb, ida.first.fhg.de
% updated by Guido Dornhege, 16/03/2005 

[T, nChans, nEpochs]= size(dat.x);
nClasses= size(dat.y,1);
if nClasses>2
  error('use proc_multicsp for more than two classes');
end

if ~exist('COV_AVG', 'var') | isempty(COV_AVG), COV_AVG=1; end
if ~exist('NORMALIZE', 'var') | isempty(NORMALIZE), NORMALIZE=0; end

R = zeros(nChans, nChans, nClasses);
if COV_AVG,
  for t= 1:nClasses,
    C= zeros(nChans, nChans);
    idx = find(dat.y(t,:)>0);
    for m= idx,
      C= C + cov(dat.x(:,:,m));
    end
    R(:,:,t)= C/length(idx);
  end
else
  for t= 1:nClasses,
    idx= find(dat.y(t,:)>0);
    x= permute(dat.x(:,:,idx), [1 3 2]);
    x= reshape(x, T*length(idx), nChans);
    R(:,:,t) = cov(x);
  end
end
if NORMALIZE,
  for t= 1:nClasses,
	 	R(:,:,t)= R(:,:,t)/trace(R(:,:,t));
  end
end

[U,D]= eig(sum(R,3)); 
P= diag(1./sqrt(diag(D)))*U';
[B,D]= eig(P*R(:,:,2)*P');
W= B'*P;	
if ~exist('nComps', 'var') | isempty(nComps),
  fi= 1:nChans;
  la= diag(D);
else
  [dd,di]= sort(diag(D));
  fi= [di(1:nComps); di(end:-1:nChans-nComps+1)];
  la= [1-dd(1:nComps); dd(end:-1:nChans-nComps+1)];
end
Wp= W(fi,:)';

dat = proc_linearDerivation(dat,Wp);

