%
function [divTr, divTe, nPick]= sampleDivision(g, nDivisions, nPick, setSize)
  
nClasses= size(g,1);
nEpochs= size(g,2);

if nargin < 4
  setSize=1;
end

if ~exist('nPick','var') || nPick > nEpochs/setSize,
  nPick= nEpochs/setSize;
end

if(nDivisions==1 || nDivisions>nEpochs/setSize),       %% leave-one-out
  nDivisions = nEpochs/setSize;
  msg= ['#folds greater than #samples / setSize\n' ...
        'switching to leave-one-out'];
  bbci_warning(msg, 'sample', mfilename);
end


divTr= cell(nDivisions, 1);
divTe= cell(nDivisions, 1);


ci=[1:nEpochs];
startidx=randperm(nEpochs/setSize).*setSize-setSize+1;
idx= [];
for si=startidx
  idx= [idx si:si+setSize-1];
end

div= round(linspace(0, nPick, nDivisions+1))*setSize;
for d= 1:nDivisions,
  sec= ci(idx(div(d)+1:div(d+1)));
  for dd= 1:nDivisions,
    if dd==d,
      divTe{dd}= [divTe{dd} sec];
    else
      divTr{dd}= [divTr{dd} sec];
    end
  end
end


