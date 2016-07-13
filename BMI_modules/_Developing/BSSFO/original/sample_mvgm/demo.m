N = 100;
stdv = 5;
numMix = 2;
mu = cat(2 , [8; 13] , [13; 20]);                %(d x 1 x M)
mu = reshape( mu, [2, 1, numMix] );
sigma = cat(2 , [stdv 0; 0 stdv] , [stdv 0; 0 stdv] );   %(d x d x M)
sigma = reshape( sigma, [2 2 numMix] );
p = cat(2 , [0.5] , [0.5]);                       %(1 x 1 x M)
p = reshape( p, [1 1 numMix] );
[Z , index] = sample_mvgm(N , mu , sigma , p);

plot(Z(1 , :) , Z(2 , :) , '+')

[w, m, v, l] = em_gm( Z', 2, [], 100, 1, [] );