function fv = proc_logprob(fv,varargin)
% fv = proc_logprob(fv,{mu1,Sigma1,mu2,Sigma2,...})
%
% get the classwise logprobability for each trial.
% If muN and SigmaN are given, they are taken as reference,
% otherwise samples are compared to the averaged class mean
% and covariances.
%
% IN: fv    - feature vector struct
%     muN   - mean for class N
%     SigmaN- covariance for class N
% OUT:fv    - feature vector struct with new field .logprob

% kraulem 08/05
if length(varargin)>0
  % class covariances and means are given
  clparams = varargin;
else
  % calculate the means and covs by hand
  clparams = {};
  for ii = 1:size(fv.y,1)
    fv1 = proc_selectClasses(fv,fv.className{ii});
    clparams{2*ii-1} = mean(fv1.x,3);
    clparams{2*ii} = cov(squeeze(fv1.x)');
  end
end

fv.log = zeros(size(fv.y));
for ii = 1:size(fv.y,1)
  % enter the logprobability for each class
  fv1 = proc_selectClasses(fv,fv.className{ii});
  fv.log(ii,find(fv.y(ii,:))) = normal_prob(squeeze(fv1.x),clparams{2*ii-1},clparams{2*ii});
end
return

function p = normal_prob(x, m, C)
% adapted from BNT (Murphy), gaussian_prob.
if length(m)==1 % scalar
  x = x(:)';
end
[d N] = size(x);
m = m(:);
M = m*ones(1,N); % replicate the mean across columns
denom = (2*pi)^(d/2)*sqrt(abs(det(C)));
mahal = sum(((x-M)'*inv(C)).*(x-M)',2);   
if any(mahal<0)
  warning('mahal < 0 => C is not psd')
end
p = -0.5*mahal - log(denom);
return