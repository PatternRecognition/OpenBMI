function [filt_b, filt_a]= butters(N,w)

nFilts= size(w,1);
if length(N)==1,
  N= N*ones(nFilts,1);
end

filt_a= cell(1, nFilts);
filt_b= cell(1, nFilts);
for ii= 1:nFilts,
  [filt_b{ii}, filt_a{ii}]= butter(N(ii), w(ii,:));
end
