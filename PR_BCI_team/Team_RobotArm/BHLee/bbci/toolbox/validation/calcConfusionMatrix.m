function [mc, me, ms, lm]= calcConfusionMatrix(y, outy, loss)
%[mc, me, ms, lm]= calcConfusionMatrix(dat, outy, <loss>)
%
% IN   dat  - data structure including label field '.y'
%      outy - output of some classifier as given by doXvalidation
%      loss - loss matrix for penelizing misclassifications
%
% OUT  mc   - confusion matrix, mc(i,j) counts the number of trials of 
%             actual class j that were classified as class i
%      me   - confusion matrix mean, me(i,j) is the mean of the % of trials
%             of actual class j that were classified as class i
%      ms   - confusion matrix std, ms(i,j) is the std of the % of trials
%             of actual class j that were classified as class i
%      lm   - classification loss matrix when misclassifying true class j
%             to class i are penelized by loss(i,j)
%
% SEE  doXvalidation, doXvalidationPlus


if ~isempty(find(outy<1 | round(outy)~=outy)),
  outy= 0.5*sign(outy)+1.5;
end

if isstruct(y), y= y.y; end

nClasses= size(y, 1);
mc= zeros(nClasses, nClasses);
me= zeros(nClasses, nClasses);
ms= zeros(nClasses, nClasses);

for ii= 1:nClasses,
  iC= find(y(ii,:));
  for jj= 1:nClasses,
    mc(jj,ii)= sum(sum(outy(:,iC)==jj));
    mme= mean(outy(:,iC)==jj, 2);
    me(jj,ii)= mean(100*mme);
    ms(jj,ii)= std(100*mme); 
  end
end

if nargout>3,
  if ~exist('loss','var') | isempty(loss),
    loss= ones(nClasses,nClasses) - eye(nClasses);
  end
  lm= (mc.*loss) ./ repmat(sum(mc,2), 1, nClasses);
end
