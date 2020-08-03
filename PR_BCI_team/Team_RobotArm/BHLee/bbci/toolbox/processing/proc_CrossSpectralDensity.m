function [fv,nWin]=proc_CrossSpectralDensity(epo, nfft, varargin)
% proc_CrossSpectralDensity - Cross spectral density estimation
% [fv,nWin]=proc_CrossSpectralDensity(epo, nfft, varargin)

% ryotat

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, struct('ovlp', 'min', 'nWin', [], 'spec', 0));

[T,d,n]=size(epo.x);
fv=epo;


step = 0;
if ~exist('nfft','var') | isempty(nfft) | nfft>T
  nfft = T;
  nWin = 1;
else
  if isempty(opt.nWin)
    if strcmp(opt.ovlp, 'min')
      % minimum overlap
      nWin = ceil(T/nfft);
    else
      % 50% overlap
      nWin = ceil(2*T/nfft-1);
    end
  else
    nWin = opt.nWin;
  end
  if nWin>1
    step = (T-nfft)/(nWin-1);
  else
    step = 0;
  end
end

if opt.spec
  fv.x=zeros(nfft, d, n);
else
  fv.x=zeros(d, d, nfft, n);
end

Sh=zeros(nfft, d, n, nWin);

idx=1:nfft;
for i=1:nWin
  iidx = ceil(idx); iidx=iidx-(iidx(end)>T);
  Sh(:, :, :, i) = fft(epo.x(iidx,:,:), nfft);
  idx = idx+step;
end

if opt.spec
  fv.x = mean(abs(Sh).^2, 4);
else
  for t=1:nfft
    for k=1:n
      fv.x(:,:,t, k) = mom2(squeeze(Sh(t,:,k,:)));
    end
  end
end

fv.t = (0:nfft-1)/nfft*fv.fs;
fv.xUnit = 'Hz';
fv.yUnit = 'power';

% second order moment of d*nSample matrix X
% the order has been changed to be consistent with complex X
function M2 = mom2(X)
M2 = (X*X')/size(X,2);
