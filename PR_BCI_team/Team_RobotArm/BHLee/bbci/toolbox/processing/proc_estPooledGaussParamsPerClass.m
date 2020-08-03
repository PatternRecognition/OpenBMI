function X = proc_estPooledGaussParamsPerClass(epo, n_pools)
%Compute estimates of mean and covariance for each class over pooled epochs.
%
%usage
% X = proc_estPooledGaussParamsPerClass(epo, { n_pools: min(sum(epo.y, 2)) })
%
%input
%  epo        Epoched data.
%  n_pools    Optional: number of pools (same for each class), default: number of epochs 
%
%output
%  X          Nested cell array (three levels): 
%               1. number of classes: size(epo.y, 1)
%               2. number of pools (n_pools), 
%               3. parameter tuple: { mean, covariance matrix }
%
%author
%  buenau@cs.tu-berlin.de

[T, n_channels, total_n_epochs] = size(epo.x);
n_classes = size(epo.y, 1);
n_epochs = sum(epo.y, 2);

if ~exist('n_pools', 'var') || isempty(n_pools), n_pools = min(n_epochs); end
if n_pools > min(n_epochs), error('Number of pools exceeds number of epochs for at least one class.'); end

X = cell(1, n_classes);

% Loop over classes.
for j=1:n_classes
  % Compute epoch-indices of current class.
  ii = find(epo.y(j,:) > 0);

  k = 1;
  psize = floor(n_epochs(j)/n_pools);
  remfrac = rem(n_epochs(j), n_pools)/n_pools;
  fracsum = 0;

  P = cell(1, n_pools);

  % Loop over pools. 
  for i=1:n_pools
    fracsum = fracsum + remfrac;
    % Pick up numerical inaccuracies in the last pool.
    csize = psize + double( (fracsum >=1) || (i == n_pools) ); 

    mu = zeros(n_channels, 1);
    covmat = zeros(n_channels, n_channels);

    % Loop over epochs in current pool.
    for l=1:csize
      % Sum up estimates of mean & covariance matrix.
      mu = mu + mean(squeeze(epo.x(:,:,ii(k+l-1))))';
      covmat = covmat + cov(squeeze(epo.x(:,:,ii(k+l-1))));
    end
    % Normalize by pool size.
    P{i} = { mu./csize, covmat./csize };

    if fracsum >= 1, fracsum = fracsum - 1; end
    k = k + csize;
  end

  X{j} = P;
end
