function est= out2label(out)
%est= out2label(out)
%
% convert classifier output to estimated labels.
%
% IN  out  - classifier output, the format can either be
%            (1) [nClasses nSamples] where each entry in one column reflects
%            membership (e.g. as probability), or
%            (2) [1 nSamples] (two-class cases only) where negative values
%            represent class 1 and positive values represent class 2.
%
% OUT est  - estimated labels

sz= size(out);
est= zeros([1 sz(2:end)]);
if size(out,1)==1,
  est(:,:)= 1.5 + 0.5*sign(out(:,:));
else
  [dummy, est(:,:)]= max(out(:,:));
end

est= permute(est, [3 2 1]);
