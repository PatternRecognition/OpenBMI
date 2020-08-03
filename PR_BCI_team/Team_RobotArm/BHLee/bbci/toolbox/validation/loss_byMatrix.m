function loss= loss_byMatrix(label, out, loss_matrix)
%loss= loss_byMatrix(label, out, loss_matrix)
%
% IN  label       - vector of true class labels (1...nClasses)
%     out         - vector of classifier outputs
%     loss_matrix - nClasses x nClasses matrix, such that loss(t,e)
%                   defines the loss for true class #t and estimated class #e.
%
% OUT loss        - vector of loss values
%
% SEE out2label

est= out2label(out);
lind= label2ind(label);

sz= size(loss_matrix);

ind = find(lind==0);
lind(ind)=[];
est(ind)=[];
loss= loss_matrix(sub2ind(sz, lind, est));
