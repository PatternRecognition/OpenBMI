function loss= loss_0_1(label, out)
%loss= loss_byMatrix(label, out)
%
% IN  label - maxtrix of true class labels, size [nClasses nSamples]
%     out   - matrix (or vector for 2-class problems)
%             of classifier outputs
%                   
% OUT loss  - vector of 0-1 loss values
%
% SEE xvalidation, loss_byMatrix, out2label

if ~islogical(label),
  label = sign(label);
end

nClasses= max(2, size(out, 1));

loss_matrix= ones(nClasses,nClasses) - eye(nClasses);
loss= loss_byMatrix(label, out, loss_matrix);
