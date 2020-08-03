function loss= loss_0_1_percent(label, out)
%loss= loss_0_1_percent(label, out)
%
% IN  label - maxtrix of true class labels, size [nClasses nSamples]
%     out   - matrix (or vector for 2-class problems)
%             of classifier outputs
%
% OUT loss  - vector of 0-1 loss values in percent
%
% SEE xvalidation, loss_byMatrix, out2label

loss= 100*loss_0_1(label, out);
