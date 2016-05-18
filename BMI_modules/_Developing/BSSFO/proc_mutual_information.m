function mi = cal_mutual_information( f1, f2, kernelWidth )
% Input:
%       f1 - training data of class 1
%       f2 - training data of cless 2
%       kernelWidth - common kernel width
% Output:
%       mi - mutual information

% avoidUnderflow = 1e-100;

% Entropy of data A: H(A)
estimatedDensity = proc_ParzenKDE( [f1'; f2'], [f1'; f2'], kernelWidth );
entropy = -sum(log(estimatedDensity)) / length(estimatedDensity);
% entropy = -sum(log(estimatedDensity+avoidUnderflow).* estimatedDensity);

% H(A|C) - class conditional entropy
classOneDensity = proc_ParzenKDE( f1', f1', kernelWidth );
classTwoDensity = proc_ParzenKDE( f2', f2', kernelWidth );
% classOneDensity = myParzenKDE( f1', [f1'; f2'], kernelWidth );
% classTwoDensity = myParzenKDE( f2', [f1'; f2'], kernelWidth );

HACOne = -sum(log(classOneDensity)) / length(classOneDensity);
HACTwo = -sum(log(classTwoDensity)) / length(classTwoDensity);
% HACOne = -sum(log(classOneDensity+avoidUnderflow).* classOneDensity);
% HACTwo = -sum(log(classTwoDensity+avoidUnderflow).* classTwoDensity);

condEntropy = (HACOne + HACTwo)/2;

mi = entropy - condEntropy;