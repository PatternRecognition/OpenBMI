function [dat, Wp, la]= proc_csp_regularised(dat, nComps, C,COV_AVG, NORMALIZE)
%[dat, W, la]= proc_csp(dat, <nComps=nChans/nClasses, C=0,COV_AVG=1, NORMALIZE=0>);
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
%      C         - the regularisation constant (between 0 and 1)
%      COV_AVG   - average covariance matrices? default 1
%      NORMALIZE - normalize covariance matrices? default 0
%
% OUT  epo       - updated data structure
%      W         - CSP projection matrix
%      la        - eigenvalue score of CSP projections (rows in W)
%
% SEE demos/classification_csp

% bb, ida.first.fhg.de


y= dat.x;
z= dat.y;
[T, nChans, nMotos]= size(y);

if ~exist('COV_AVG', 'var') | isempty(COV_AVG), COV_AVG=1; end
if ~exist('NORMALIZE', 'var') | isempty(NORMALIZE), NORMALIZE=0; end

if size(z,1)==1
  z= [z<0; z>0];
end

if COV_AVG
  for t= 1:size(z,1),
    R= zeros(nChans);
    for m= find(z(t,:) >0),
      R= R + cov(y(:,:,m));
    end
    RR{t}= R/nChans;
  end
else
  for t= 1:size(z,1),
    cl= find(z(t,:) >0);
    x= permute(y(:,:,cl), [1 3 2]);
    x= reshape(x, T*length(cl), nChans);
    RR{t} = cov(x);
  end
end
R = zeros(nChans,nChans, size(z,1));
for t = 1:size(z,1),
 	if NORMALIZE,
	 	R(:,:,t)= RR{t}/trace(RR{t});
 	else
		R(:,:,t) = RR{t};
	end
end

if C>0
  d = size(R,1);
  for t = 1:size(R,3);
    R(:,:,t) = (1-C)*R(:,:,t)+ C/d*trace(R(:,:,t))*eye(d);
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


yo= zeros(size(y,1), size(Wp,2), nMotos);
for m= 1:nMotos,
  yo(:,:,m)= y(:,:,m)*Wp;
end

dat.x= yo;
dat.origClab= dat.clab;

k= 0;
dat.clab= cell(1, size(Wp,2));
for ii= 1:size(z,1), 
  for jj= 1:size(Wp,2)/size(z,1),
    k= k+1;
    if isfield(dat, 'className'),
      dat.clab{k}= sprintf('%s:csp%d', dat.className{ii}, jj);
    else
      dat.clab{k}= sprintf('cl%d:csp%d', ii, jj);
    end
  end
end
