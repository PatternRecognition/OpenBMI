function [nTrials, nDivisions, nPick]= getValidationTriple(nTrialsIn)
%[nTrials, nDivisions, nPick]= getValidationTriple(nTrialsIn)

nTrials= nTrialsIn(1);

if length(nTrialsIn)>1,
  nDivisions= nTrialsIn(2);
else
  nDivisions= 5;
end

if length(nTrialsIn)>2,
  nPick= [nTrialsIn(3:end)];
else
  nPick= [];
end
