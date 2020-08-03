function loss= loss_classwiseNormalized(label, out, N)
%loss= loss_classwiseNormalized(label, out, N)
%
% IN  label - vector of true class labels (1...nClasses)
%     out   - vector of classifier outputs
%     N     - vector of length nClasses, the i-th element specifying the
%             number of samples contained in class i in the whole database
%
% OUT loss  - vector of loss values
%
% Typically this function is used in xvalidation using
%   xvalidation(fv, model, 'loss', {'classwiseNormalized',sum(fv.y,2)});
% which gives the correct value for input argument N.
% 
% SEE out2label

nClasses= size(label, 1);
est= out2label(out);
lind= label2ind(label);

if nargin<3 || isempty(N),
  %% this estimates class sizes on the validation set
  N= sum(label, 2);
else
  N= reshape(N, [nClasses 1]);
end

%% problem when any(N==0)
loss_matrix= (sum(N)./N)/nClasses * ones(1,nClasses);
loss_matrix= loss_matrix - diag(diag(loss_matrix));

loss= loss_matrix(sub2ind([nClasses nClasses], lind, est));
