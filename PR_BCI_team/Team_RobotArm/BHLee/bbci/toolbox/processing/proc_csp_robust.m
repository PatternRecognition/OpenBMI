function [dat, Wp, la]= proc_csp_robust(dat, nComps, typ, COV_AVG, NORMALIZE)
%[dat, W, la]= proc_csp(dat, <nComps=nChans/nClasses, typ = 'norm_1',COV_AVG=1, NORMALIZE=0>);
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
% extended to multi-class case by guido


y= dat.x;
z= dat.y;
[T, nChans, nMotos]= size(y);

if ~exist('COV_AVG', 'var') | isempty(COV_AVG), COV_AVG=0; end
if ~exist('NORMALIZE', 'var') | isempty(NORMALIZE), NORMALIZE=0; end

if size(z,1)==1
  z= [z<0; z>0];
end

if COV_AVG
  for t= 1:size(z,1),
    R= zeros(nChans);
    for m= find(z(t,:)>0),
      if ischar(typ)
        R= R + estimate_robust_covariance(y(:,:,m)',typ);
      else
        R = R + robust_covariances_least_informative(y(:,:,m)',typ);
      end
    end
    RR{t}= R/nChans;
  end
else
  for t= 1:size(z,1),
    cl= find(z(t,:) >0);
    x= permute(y(:,:,cl), [1 3 2]);
    x= reshape(x, T*length(cl), nChans);
    if ischar(typ)
      RR{t} = estimate_robust_covariance(x',typ);
    else
      RR{t} = robust_covariances_least_informative(x',typ);
    end
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

[U,D]= eig(sum(R,3)); 
P= diag(1./sqrt(diag(D)))*U';
if size(z,1)==2
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
else
	if ~exist('nComps','var') | isempty(nComps)
 		[B,D]= eig(P*R(:,:,1)*P');
 		Wp= B'*P;
    la= diag(D);
	else
		Wp= [];
    la= [];
		for t = 1:size(z,1),
 			[B,D]= eig(P*R(:,:,t)*P');
 			W = B'*P;
 			[dd,di]= sort(-diag(D));
      W = W(di(1:nComps),:);
      Wp = [Wp;W];
      la = [la;-dd(1:nComps)];
		end  
	 	Wp = Wp';
	end
end


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
