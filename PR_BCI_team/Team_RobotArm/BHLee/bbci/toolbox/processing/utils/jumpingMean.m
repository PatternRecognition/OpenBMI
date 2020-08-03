function xm= jumpingMean(x, nSamples)

[T, N]= size(x);
nMeans= floor(T/nSamples);
xm= permute(mean(reshape(x((T-nMeans*nSamples+1):T,:,:), ...
												 [nSamples,nMeans,N]),1),[2 3 1]);
