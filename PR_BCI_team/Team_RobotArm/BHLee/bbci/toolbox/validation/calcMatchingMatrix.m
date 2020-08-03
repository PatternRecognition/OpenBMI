function [mc, me, ms, lm]= calcMatchingMatrix(y, outy, loss)
%[mc, me, ms, lm]= calcMatchingMatrix(dat, outTe, <loss>)
%
% OBSOLETE please use calcConfusionMatrix instead,
%          but watch out as outputs are transposed!
%
% IN   dat  - data structure including label field '.y'
%      outy - output of some classifier as given by doXvalidation
%      loss - loss matrix for penelizing misclassifications
%
% OUT  mc   - mismatching matrix, mc(i,j) counts the number of trials of class i
%             that were classified as class j
%      me   - mismatching matrix, me(i,j) is the mean of the % of trials
%             of class i that were classified as class j
%      ms   - mismatching matrix, ms(i,j) is the std of the % of trials
%             of class i that were classified as class j
%      lm   - classification loss matrix when misclassifying true class i
%             to class j are penelized by loss(i,j)
%
% SEE  doXvalidation

if isstruct(y), y= y.y; end
if ~any(ismember(outy, [-1 1])),
  outy(:)= 1.5 + 0.5*sign(outy(:));
end

nClasses= size(y, 1);
mc= zeros(nClasses, nClasses);
me= zeros(nClasses, nClasses);
ms= zeros(nClasses, nClasses);

for ii= 1:nClasses,
  iC= find(y(ii,:));
  for jj= 1:nClasses,
    mc(ii,jj)= sum(sum(outy(:,iC)==jj));
    mme= mean(outy(:,iC)==jj, 2);
    me(ii,jj)= mean(100*mme);
    ms(ii,jj)= std(100*mme); 
  end
end

if nargout>3,
  if ~exist('loss','var') | isempty(loss),
    loss= ones(nClasses,nClasses) - eye(nClasses);
  end
  lm= (mc.*loss) ./ repmat(sum(mc,2), 1, nClasses);
end
