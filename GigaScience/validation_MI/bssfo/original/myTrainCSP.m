function [W, D] = myTrainCSP( X1, X2 )

[nc1, ns1, nt1] = size( X1 );
[nc2, ns2, nt2] = size( X2 );

XX1 = reshape( permute(X1, [2 3 1]), [ns1*nt1, nc1]);
S1 = cov(XX1(:,:));

XX2 = reshape( permute(X2, [2 3 1]), [ns2*nt2, nc2]);
S2 = cov(XX2(:,:));

[W,D] = eig(S1, S1+S2);
