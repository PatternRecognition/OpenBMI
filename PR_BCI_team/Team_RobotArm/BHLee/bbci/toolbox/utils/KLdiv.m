function out = KLdiv(mu1,mu2,Sig1,Sig2)
% Kullback Leibler Divergence of two Gaussian distributions.

% kraulem 09/05

% make column vectors:
mu2 = mu2(:)-mu1(:);

% cosmetic code: don't recalculate Sig inversion twice.
Sig2_inv = inv(Sig2);
Sig1_Sig2_inv = Sig1*Sig2_inv;

out = log(det(Sig1_Sig2_inv)) + trace(eye(length(mu1))-Sig1_Sig2_inv) ...
      - mu2'*Sig2_inv*mu2;
out = -.5*out;
return

