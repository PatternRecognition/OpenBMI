function loss= loss_negMI(label, out)
%loss= loss_negMI(label, out)
%
% IN  label - vector of true class labels (1...nClasses)
%     out   - array of classifier outputs
%
% OUT loss  - negative MI value 
%
% NOTE:
% the author (stl) is fully aware of the fact that MI is not a loss 
% function in the strict sense of the definition, consequently its
% negative value is used. 
% 
% MI is only implemented for BINARYy classification problems 
%
%
% SEE roc_curve

nClasses= size(label, 1);

if nClasses > 2, 
  error('Loss function: negMI only implemented for binary classification tasks!!!\n') ; 
end ;

valid= find(all(~isnan(out),1));


c = [1:nClasses]*label(:,valid) ;
d = out(:,valid) ;

c=c(:).';
ntr=length(c);
[nr,nc]=size(d);

CL=unique(c);
if length(CL)~=2,
        error('invalid class labels: just one class label given');
end;

% time course of the SNR+1, Mutual Information, and the Error rate [1]
VAR1    = std(d,1,2).^2 
VAR2    = std(d(:,c==CL(1)),1,2).^2 + std(d(:,c==CL(2)),1,2).^2 

SNRp1(VAR2~=0) 	= 2*VAR1(VAR2~=0)./VAR2(VAR2~=0) ;
SNRp1(VAR2==0) = Inf;

loss 	= -log2(SNRp1)/2;

