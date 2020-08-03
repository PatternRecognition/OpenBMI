function [fv, W, la] = proc_cspSpec(fv, nof, F, nfft, output)
% proc_cspSpec - CSP for spectral weighting F.
% [fv, W, la] = proc_cspSpec(fv, nof, F, nfft, output)
% [fv, W, la] = proc_cspSpec(fv, nof, F, fvCS, output) % use the precomputed spectrum.

% ryotat

if ~exist('output','var'), output=0; end

ncls = 2;
[T,d,n] = size(fv.x);
nff = size(F,2);

if isstruct(nfft)
  if ~isequal(fv.title, nfft.title)
    error('mismatched title!');
  end
  
  CS1=nfft.x(:,:,:,1);
  CS2=nfft.x(:,:,:,2);
else
  fprintf('Calculating the cross spectrum...\n');
  fvCS = proc_averageCSD(fv, nfft, 'ovlp', 'min');
  CS1 = fvCS.x(:,:,:,1);
  CS2 = fvCS.x(:,:,:,2);
end

W  = zeros(d, ncls*nof, nff);
la = zeros(1, ncls*nof, nff);

for i=1:nff
  V1 = apply_temporal_filter(CS1, F(:,i));
  V2 = apply_temporal_filter(CS2, F(:,i));

  [W(:,:,i), la(:,:,i)] = csp(V1,V2,nof);
end

switch(output)
 case 1
  fv = proc_linearDerivation(fv, reshape(W, [d,ncls*nof*nff]));
 case 2
  fv = fvCS;
end

function V = apply_temporal_filter(CS, F)
d = size(CS, 1);
V=zeros(d,d);
for i=1:d, V(:,i)=real(squeeze(CS(i,:,:))*F)/size(F,1); end

function [W, la] = csp(V1, V2, nof)
d = size(V1,1);

[EVP, EDP] = eig(V1+V2);
P = inv(sqrt(EDP))*EVP';
[EV2, ED2] = eig(P*V2*P');
Q = EV2'*P;

[es, is] = sort(real(diag(ED2))');

if ~isempty(nof)
  iis = [1:nof, d:-1:d-nof+1];
else
  iis = 1:d;
end

W = Q(is(iis),:)';
la = es(iis);

