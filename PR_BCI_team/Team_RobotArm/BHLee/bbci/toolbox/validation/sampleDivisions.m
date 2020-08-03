function [divTr, divTe, nPick]= sampleDivisions(g, xTrials, equi, test)
%[divTr, divTe]= sampleDivisions(goal, xTrials, <equi, test>)
% test describes where test trials can only come from (default all)

if ~exist('test','var') || isempty(test)
  test = 1:size(g,2);
end

if length(xTrials)>=3 && xTrials(3)<0,
%% set xTrials(3) to the size of the training set
  nValid= sum(any(g));
  if isequal(xTrials(1:2),[1 1]),  %% leave-one-out
    xTrials(3)= nValid-1;
  else
    xTrials(3)= round((xTrials(2)+xTrials(3))/xTrials(2)*nValid);
  end
end

nTrials= xTrials(1);
nDivisions= xTrials(2);
if length(xTrials)>2,
  nPick= xTrials(3:end);
else
  nPick= 0;
end

if exist('equi','var') && ~isempty(equi),
  nPick0= nPick;
  divTr= cell(nTrials, 1);
  divTe= cell(nTrials, 1);
  for ti= 1:nTrials,
    subs= chooseEquiSubset(equi);
    goal= zeros(1, size(g,2));
    for si= 1:length(subs),
      goal(si, subs{si})= 1;
    end
    [dTr, dTe, nPick]= sampleDivisions(goal, [1 nDivisions nPick0]);
    divTr{ti}= dTr{1};
    divTe{ti}= dTe{1};
  end
  return
end

nClasses= size(g,1);
nEventsInClass= zeros(1, nClasses);
for cl= 1:nClasses,
  nEventsInClass(cl)= length(find(g(cl,test)));
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

if nDivisions<=0,
  nPick= nPick + nDivisions;
  idxTr= [];
  idxTe= [];
  for ci= 1:nClasses,
    clind= find(g(ci,:))';
    [idx coIdx]= chooseSubsets(nEventsInClass(ci), nPick(ci), nTrials);
    idxTr= [idxTr clind(idx)];
    idxTe= [idxTe clind(coIdx)];
  end
  %divTr= num2cell(idxTr, 2);
  %divTe= num2cell(idxTe, 2);
  for ti= 1:nTrials,
    divTr{ti}= {idxTr(ti,randperm(size(idxTr,2)))};
    divTe{ti}= {idxTe(ti,randperm(size(idxTe,2)))};
  end
  
else

  divTr= cell(nTrials,1);
  divTe= cell(nTrials,1);
  for ti= 1:nTrials,
    [divTr{ti}, divTe{ti}]= sampleDivision(g(:,test), nDivisions, nPick);
  end
end

testrest = setdiff(1:size(g,2),test);

if ~isempty(testrest)
  divTe = changedivTe(divTe,test);
  divTr = changedivTe(divTr,test);
  divTr = adddivTr(divTr,testrest);
end


%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION
%%%%%%%%%%%%%%%%%%%%%%%


function divTe = changedivTe(divTe,test)

if iscell(divTe)
  for i = 1:length(divTe)
    divTe{i} = changedivTe(divTe{i},test);
  end
else
  divTe = test(divTe);
end


function divTr = adddivTr(divTr,test)

if iscell(divTr)
  for i = 1:length(divTr)
    divTr{i} = adddivTr(divTr{i},test);
  end
else
  divTr= [divTr,test];
end
