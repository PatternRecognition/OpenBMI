function [fv, W, F, la, Whist, Fhist, lahist] = proc_iterCspSpec(epo, nStep, nof, nfft, band, p, q)
% [fv, W, F, la] = proc_iterCspSpec(epo, nStep, nof, nfft, band, p, q)
%
% NOTE: p and q are already reparameterized.
%       one step contains one spatial update and one spectral update

% ryotat

[T,d,n]=size(epo.x);
ncls = 2;

% initial spectral filter
F = ones(nfft, 1);

Whist = zeros(d, ncls*nof, nStep);
Fhist = zeros(nfft, ncls*nof, nStep);
lahist = zeros(nStep, ncls*nof);

% precompute cross spectrum
fprintf('Calculating the cross spectrum...\n');
fvCS = proc_averageCSD(epo, nfft, 'ovlp', 'min');

for n=1:nStep
  % start from CSP
  [fv, W, la] = proc_cspSpec(epo, nof, F, fvCS, 0);

  % choose the best
  if size(W,3)>1
    [dum, ix1] = min(la(1,1,:));
    [dum, ix2] = max(la(1,nof+1,:));
    
    W  =  [W(:,1:nof,ix1), W(:,nof+(1:nof),ix2)];
    la = [la(1,1:nof,ix1),la(1,nof+(1:nof),ix2)];
  else
    % for the initial filter
    F = repmat(F, [1,size(W,2)]);
  end
  
  % spectral update (reparametrized!)
  [fv, F] = proc_opttempo(epo, W, nfft, band, p+q, q);
  F = [F(:,1:nof,1), F(:,nof+(1:nof),2)];
  
  Whist(:,:,n) = W;
  Fhist(:,:,n) = F;
  lahist(n,:)  = la;
end

fv = proc_log_project_cssp2(epo, W, F);
