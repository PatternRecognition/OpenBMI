function loss= loss_multinom_doubleTarget(label, out, trialSize)
%loss= mulitnom(label, out)
%
% IN  label - matrix of true class labels, size [nClasses nSamples]
%     out   - matrix (or vector for 2-class problems)
%             of classifier outputs
%                   
% OUT loss  - vector of mulitnomial loss values
%
% SEE xvalidation, loss_byMatrix, out2label

  
l=reshape(label(2,:), trialSize, []);
o=reshape(min(out,[],1), trialSize, []);

[os oi(1,:)]=min(o(1:6,:));
[os oi(2,:)]=min(o(7:11,:));
oi(2,:)=oi(2,:)+6;
[ls li(1,:)]=min(l(1:6,:));
[ls li(2,:)]=min(l(7:11,:));
li(2,:)=li(2,:)+6;

correctTrials=(sum(li==oi,1)==2);
loss = 1-mean(correctTrials);
