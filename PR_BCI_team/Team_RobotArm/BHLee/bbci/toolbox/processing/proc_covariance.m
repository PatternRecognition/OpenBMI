function epo = proc_covariance(epo);
% proc_covariance - calculate the covariance between channels for each sample
%
% epocv = proc_covariance(epo)
%
% see also:
%  proc_variance

[T,d,n]=size(epo.x);

V = zeros(1,d*d,n);
for i=1:n
  V(1,:,i) = reshape(cov(epo.x(:,:,i)),[1,d*d]);
end
epo.x = V;
if isfield(epo,'t')
  epo.t = epo.t(end);
end

if isfield(epo,'clab')
  clab=cell(1,d*d);
  ii=1;
  for i=1:d, for j=1:d,
      clab{ii}=[epo.clab{i}, ' x ' epo.clab{j}];
      ii=ii+1;
    end, end
    
    epo.clab = clab;
end
