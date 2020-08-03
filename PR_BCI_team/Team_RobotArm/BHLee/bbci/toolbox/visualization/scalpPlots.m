function H= scalpPlots(mnt, W, varargin)

nPat= size(W,2);
for n= 1:nPat,
  suplot(nPat, n);
  H(n)= scalpPlot(mnt, W(:,n), varargin{:});
end
