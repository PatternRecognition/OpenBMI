function [mu,C] = proc_estPooledGaussParams(epo, n_pools)
%Compute estimates of mean and covariance over pooled epochs.
%
%usage
% [mu, C] = proc_estPooledGaussParams(epo, { n_pools: min(sum(epo.y, 2)) })
%
%input
%  epo        Epoched data.
%  n_pools    Optional: number of pools (same for each class), default: number of epochs 
%
%output
%
%author
%  buenau@cs.tu-berlin.de

[T, n_channels, n_epochs] = size(epo.x);

if ~exist('n_pools', 'var') || isempty(n_pools), n_pools = n_epochs; end
if n_pools > n_epochs, error('Number of pools exceeds number of epochs for at least one class.'); end

R_mu = zeros(n_channels, n_pools);
R_C = zeros(n_channels, n_channels, n_pools);

  k = 1;
  psize = floor(n_epochs/n_pools);
  remfrac = rem(n_epochs, n_pools)/n_pools;
  fracsum = 0;

  % Loop over pools. 
  for i=1:n_pools
    fracsum = fracsum + remfrac;
    % Pick up numerical inaccuracies in the last pool.
    csize = psize;
    if remfrac > 0
      csize = csize + double( (fracsum >=1) || (i == n_pools) ); 
    end

    mu = zeros(n_channels, 1);
    covmat = zeros(n_channels, n_channels);

    % Loop over epochs in current pool.
    for l=1:csize
      % Sum up estimates of mean & covariance matrix.
      mu = mu + mean(squeeze(epo.x(:,:,(k+l-1))))';
      covmat = covmat + cov(squeeze(epo.x(:,:,(k+l-1))));
    end
    % Normalize by pool size.
    R_mu(:,i) = mu./csize;
    R_C(:,:,i) = covmat./csize;

    if fracsum >= 1, fracsum = fracsum - 1; end
    k = k + csize;
  end

mu = R_mu;
C = R_C;
