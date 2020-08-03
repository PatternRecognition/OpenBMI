function loss= multinom(label, out, nClasses)
%loss= mulitnom(label, out)
%
% IN  label - maxtrix of true class labels, size [nClasses nSamples]
%     out   - matrix (or vector for 2-class problems)
%             of classifier outputs
%                   
% OUT loss  - vector of mulitnomial loss values
%
% SEE xvalidation, loss_byMatrix, out2label

[mmtrue, ixtrue] = max(reshape(label(2, :), nClasses, []));
[mm, ix] = max(reshape(out(end, :), nClasses, []));

loss = 1-mean(ixtrue==ix);
