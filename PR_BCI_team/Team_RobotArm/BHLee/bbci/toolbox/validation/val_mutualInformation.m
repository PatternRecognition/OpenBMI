function mi = val_mutualInformation(out,labels);
%val_mutualInformation calculates the mutual information between the 
%continuous output signal out and the true labels as suggested by Schlögl 
%et al (2002)
%
% usage:
%    mi = val_mutualInformation(out,labels);
%
% input:
%    out    the continuous one-dimensional classifier output
%    labels the true labels as 2 times nTrials logical array
%
% output:
%    mi     the mutual information
%
% SEE  Schlögl A., Neuper C. Pfurtscheller G., 
%      Estimating the mutual information of an EEG-based 
%      Brain-Computer-Interface. 
%      Biomedizinische Technik 47(1-2): 3-8, 2002
%
% Guido Dornhege, 17/02/2005
% $Id: val_mutualInformation.m,v 1.1 2005/02/17 09:45:10 neuro_cvs Exp $

if size(labels,1)~=2 
  error('only two class problems');
end

if length(out)~=size(labels,2)
  error('out and labels have a different number of trials');
end

mi = var(out(find(labels(1,:))))+var(out(find(labels(2,:))));
mi = 2*var(out)/mi;
mi = 0.5*log2(mi);
