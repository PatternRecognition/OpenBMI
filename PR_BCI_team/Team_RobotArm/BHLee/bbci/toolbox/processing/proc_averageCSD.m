% [fv, nWin]=proc_averageCSD(epo, nfft, varargin)
function [fv, nWin]=proc_averageCSD(epo, nfft, varargin)

% ryotat

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, struct('ovlp', 'min', 'nWin', [], 'spec', 0));

[T,d,n]=size(epo.x);
ncls=size(epo.y,1);
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
  fv.x=zeros(nfft, d, ncls);
else
  fv.x=zeros(d, d, nfft, ncls);
end

Sh=zeros(nfft, d, n, nWin);

idx=1:nfft;
for i=1:nWin
  iidx = ceil(idx); iidx=iidx-(iidx(end)>T);
  Sh(:, :, :, i) = fft(epo.x(iidx,:,:), nfft);
  idx = idx+step;
end

if opt.spec
  for k=1:ncls
    Ik=epo.y(k,:)>0;
    Shk = reshape(Sh(:,:,Ik,:), [nfft, d, sum(Ik)*nWin]);
    fv.x(:,:,k) = mean(abs(Shk).^2, 3);
  end
else
  for t=1:nfft
    for k=1:ncls
      Ik=epo.y(k,:)>0;
      fv.x(:,:,t, k) = mom2(reshape(squeeze(Sh(t,:,Ik,:)), [d, sum(Ik)*nWin]));
    end
  end
end

fv.y = eye(ncls);
fv.t = (0:nfft-1)/nfft*fv.fs;
fv.xUnit = 'Hz';
fv.yUnit = 'power';

% second order moment of d*nSample matrix X
% the order has been changed to be consistent with complex X
function M2 = mom2(X)
M2 = (X*X')/size(X,2);
