function p = bitrateinv(bit,n);

if ~exist('n','var'), n=2; end
bbci_warning('off', 'MATLAB:fzero:UndeterminedSyntax');

p = zeros(size(bit));
for i = 1:prod(size(bit));
  p(i) = fminsearch(inline('(bitrate(min(max(x,0),1),n)-y)^2','x','y','n'),1,[],bit(i),n);    end

