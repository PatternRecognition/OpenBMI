function [fv, F] = proc_opttempo(fv, W, nfft, band, p, q)
% proc_opttempo - a spectral filter for each CSP projection W
% [fv, F] = proc_opttempo(fv, W, nfft, band, p, q)
%
% see also proc_cspSpec

% ryotat

ncls = 2;

freq = (0:nfft-1)/nfft*fv.fs;
fmask = freq>=band(1) & freq<=band(2);

% Temporal Filter Optimization
%% MSK: something's fishy here, pairing of opts for proc_CSD is wrong
%spec = proc_CSD(proc_linearDerivation(fv, W),...
%            nfft, 'spec', 1, 'ovlp', 'min');
%%
spec = proc_CSD(proc_linearDerivation(fv, W));

Sw = spec.x;

F = zeros(nfft, size(W,2), ncls);

for icls=1:ncls
  ocls = ncls+1-icls;
  S1 = Sw(fmask, :, fv.y(icls,:)>0);
  S2 = Sw(fmask, :, fv.y(ocls,:)>0);

  F(fmask,:,icls) = opttempo(S1, S2, p, q);
  F(2:end,:,icls) = F(2:end,:,icls) + F(end:-1:2,:,icls);
end

function f = opttempo(S1, S2, p, q)
T = size(S1,1);

m1 = mean(S1, 3); m2 = mean(S2, 3);
v1=std(S1,[],3).^2;
v2=std(S2,[],3).^2;
d = (m1-m2)./(v1+v2); d(d<0)=0;
s = m1+m2;
f = (d.^q).*(s.^p);

ixZ = sum(f)==0;
if any(ixZ)
  f(:,ixZ) = 1/T*ones(T, sum(ixZ));
end

if any(~ixZ)
  f(:,~ixZ) = f(:,~ixZ)./(ones(T,1)*sum(f(:,~ixZ)));
end
