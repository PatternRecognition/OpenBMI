function [divTr, divTe]= sample_evenOdd(label, whichForTrain)
%[divTr, divTe]= sample_evenOdd(label, <whichForTrain>)
%
% IN  label   - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     whichForTrain - string 'even', 'odd', 'both, specifying
%               which part should be used for training. 'both'
%               generates two partitions, one with evens the other
%               with odds as training set, default 'odd'.
% 
% OUT divTr   - divTr{1}: cell array holding the training sets
%     divTe   - analogue to divTr, for the test set

if ~exist('whichForTrain','var'),
  whichForTrain= 'odd';
end

valid= find(any(label==1,1));
odd= valid(1:2:end);
even= valid(2:2:end);

switch(whichForTrain),
 case 'even',
  divTr= {{even}};
  divTe= {{odd}};
 case 'odd',
  divTr= {{odd}};
  divTe= {{even}};
 case 'both',
  divTr= {{even,odd}};
  divTe= {{odd,even}};
 otherwise,
  error('policy not known');
end
