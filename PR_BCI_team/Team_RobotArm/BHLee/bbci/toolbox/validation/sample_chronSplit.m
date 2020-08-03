function [divTr, divTe]= sample_chronSplit(label, perc)
%[divTr, divTe]= sample_chronSplit(label, perc)
%
% IN  label   - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     perc    - percentage of the training samples, default 0.5.
% 
% OUT divTr   - divTr{1}{1}: holds the training set
%     divTe   - analogue to divTr, for the test set

if ~exist('perc','var'), perc=0.5; end

valid= find(any(label==1,1));
nTrain= ceil(perc*length(valid));

divTr= {{valid(1:nTrain)}};
divTe= {{valid(nTrain+1:end)}};
