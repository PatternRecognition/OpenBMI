function [divTr, divTe,nPick]= sampDivisions(g, xTrials, equi)
%[divTr, divTe]= sampleDivisions(goal, xTrials, <equi>)

[nTrials,nTrain,nPick] = getValidationTriple(xTrials);
if isempty(nPick)
  nPick = 0;
end



if exist('equi','var') & ~isempty(equi),
  divTr= cell(nTrials, 1);
  divTe= cell(nTrials, 1);
  for ti= 1:nTrials,
    subs= chooseEquiSubset(equi);
    goal= zeros(1, size(g,2));
    for si= 1:length(subs),
      goal(si, subs{si})= 1;
    end
    [dTr, dTe]= sampDivisions(goal, [1 nTrain]);
    divTr{ti}= dTr{1};
    divTe{ti}= dTe{1};
  end
  return
end

nClasses= size(g,1);
nEventsInClass= zeros(1, nClasses);
for cl= 1:nClasses,
  nEventsInClass(cl)= length(find(g(cl,:)));
end

if nPick==0,
  nPick= nEventsInClass;
elseif length(nPick)==1,
  totalPick= nPick;
  nPick= round(totalPick*nEventsInClass/sum(nEventsInClass));
  nPick= min([nPick; nEventsInClass]);
elseif length(nPick)~=nClasses,
  error('nPick must be scalar or match #classes');
end

divTr= cell(nTrials,1);
divTe= cell(nTrials,1);

if nTrain<1 
  nTrain = floor(nTrain*nPick);
else
  nTrain = floor(nTrain*nEventsInClass/sum(nEventsInClass));
end

for ti= 1:nTrials,
  [divTr{ti}, divTe{ti}]= sampDivision(g, nTrain, nPick);
end




















