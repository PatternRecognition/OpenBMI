function [dat, Wp, la]= proc_csp_bayes(dat, nComps, COV_AVG, NORMALIZE)
%[dat, W, la]= proc_csp_bayes(dat, <nComps=nChans/nClasses, COV_AVG=1, NORMALIZE=0>);
%
% Common spatial patterns (CSP) with covariance estimates based on an
% undirected graphical model (Lauritzen)
%
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
% code for undirected graphical models by kraulem and anton

dat0 = dat;
[T, nChans, nEpochs]= size(dat.x);
nClasses= size(dat.y,1);

if ~exist('COV_AVG', 'var') | isempty(COV_AVG), COV_AVG=1; end
if ~exist('NORMALIZE', 'var') | isempty(NORMALIZE), NORMALIZE=0; end
if ~isfield(dat,'lambda'), lambda = .5;end
if ~isfield(dat,'cliques'), dat.cliques = [];end

R = zeros(nChans, nChans, nClasses);
if COV_AVG,
 for t= 1:nClasses,
  C= zeros(nChans, nChans);
  if isempty(dat.cliques)
    for m= find(dat.y(t,:)),
      C= C + cov(dat.x(:,:,m));
    end
  else
    % Estimate cov per trial, then average. This is very time-consuming!
    for m= find(dat.y(t,:)),
      C= C + covFromCliques(squeeze(dat.x(:,:,m))',dat.cliques, 'verbosity',0);
    end
  end
  R(:,:,t)= C/sum(dat.y(t,:));%/nChans;
 end
 %R = proc_get_covariances(dat);
else
  for t= 1:nClasses,
    idx= find(dat.y(t,:));
    x= permute(dat.x(:,:,idx), [1 3 2]);
    x= reshape(x, T*length(idx), nChans);
    if isempty(dat.cliques),
      R(:,:,t) = cov(x);
    else
      R(:,:,t) = covFromCliques(x', dat.cliques, 'verbosity', 0);
% $$$       origR = cov(x);
% $$$       figure;
% $$$       subplot(2,2,1);
% $$$       imagesc(origR);
% $$$       colorbar;
% $$$       title('Standard covariance');
% $$$       subplot(2,2,2);
% $$$       imagesc(R(:,:,t));
% $$$       colorbar;
% $$$       title('Covariance from cliques');
% $$$       subplot(2,2,3);
% $$$       imagesc(inv(origR));
% $$$       colorbar;
% $$$       title('Standard precision');
% $$$       subplot(2,2,4);
% $$$       imagesc(inv(R(:,:,t)));
% $$$       colorbar;
% $$$       title('Precision from cliques');
% $$$       keyboard;
    end
  end
end
if NORMALIZE,
  for t= 1:nClasses,
	 	R(:,:,t)= R(:,:,t)/trace(R(:,:,t));
  end
end
% $$$ keyboard
if isfield(dat,'prior')
  %averaging of covariance matrices.
  R = dat.lambda*R + (1-dat.lambda)*dat.prior;
end
if isfield(dat,'Sigma')
  %averaging with other experiments.
  S = zeros(nChans, nChans, nClasses);
  for l = 1:length(dat.Sigma)
    for i = 1:nClasses
      S(:,:,i) = S(:,:,i)+dat.Sigma(l).S(:,:,i)/dat.Sigma(l).n(i);
    end
  end
  S = S/length(dat.Sigma);
  R = dat.lambda*R + (1-dat.lambda)*S;
end

[U,D]= eig(sum(R,3)); 
P= diag(1./sqrt(diag(D)))*U';
if nClasses==2
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
		for t = 1:nClasses,
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

nNewChans= size(Wp, 2);
xx= zeros(T, nNewChans, nEpochs);
for m= 1:nEpochs,
  xx(:,:,m)= dat.x(:,:,m)*Wp;
end
dat.x= xx;

dat.origClab= dat.clab;
k= 0;
dat.clab= cell(1, nNewChans);
for ii= 1:nClasses, 
  for jj= 1:nNewChans/nClasses,
    k= k+1;
    if isfield(dat, 'className'),
      dat.clab{k}= sprintf('%s:csp%d', dat.className{ii}, jj);
    else
      dat.clab{k}= sprintf('cl%d:csp%d', ii, jj);
    end
  end
end
