function out= val_removeNaNs(out)
%out= val_removeNaNs(out)
%
% removes NaN values from the classifier output matrix as given by
% xvalidation (3rd output argument). NaN values are introduced, when
% not every sample was in the test set in one shuffle.
% the main task of this function is to check the input matrix for
% consistency.

[nClasses, nSamples, nShuffles]= size(out);

%% this is just to check consistency
for uu= 1:nShuffles,
  isanynan= find(any(isnan(out(:,:,uu)),1));
  isallnan= find(all(isnan(out(:,:,uu)),1));
  if ~isequal(isanynan, isallnan),
    error('inconsistency in first dimension');
  end
  if uu==1,
    nansPerShuffle= length(isallnan);
  else
    if length(isallnan) ~= nansPerShuffle,
      error('inconsistency between shuffles');
    end
  end
end

%% this does the thing
out= permute(out, [2 3 1]);
out(find(isnan(out)))= [];
out= ipermute(out, [2 3 1]);
