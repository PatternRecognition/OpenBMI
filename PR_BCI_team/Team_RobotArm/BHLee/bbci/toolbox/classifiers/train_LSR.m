function C= train_LSR(xTr, yTr, balanced)
%C= train_LSR(xTr, yTr, <balanced>)
%
% least squares regression to the labels
%
% if classes have different numbers of samples in the training
% set, you should use the 'balanced' option (i.e. pass a 1 or
% the string 'balanced' as third argument)

if exist('balanced','var') & (balanced==1 | isequal(balanced,'balanced')),
  if size(yTr,1)==1,
    error('balanced LSR does not work for one-dim goals');
  end
  N= sum(yTr, 2);
  yTr= diag(max(N)./N)*yTr;
end

if size(yTr,1)==2, yTr=[-1 1]*yTr; end 

wb= yTr*pinv([xTr; ones(1,size(xTr,2))]);
C.w= wb(:,1:end-1)';
C.b= wb(:,end);
