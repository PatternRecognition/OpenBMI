function [divTr, divTe, nPick]= sample_divisionsPlus(g, xTrials, ...
                                                     equi, test, train)
%[divTr, divTe]= valdiv_sample(goal, xTrials, <equi, test, train>)
% test describes where test trials can only come from (default all)
% train describes where training trials can only come from (default all)
%  If both test and train don't have full length, train will be set to the 
%  default value.

if ~exist('test','var') || isempty(test)
  test = 1:size(g,2);
  if ~exist('train','var') || isempty(train)
    train = 1:size(g,2);
    % restrict test samples on test.
    restrict = 0;
  else
    % restrict training samples on train.
    restrict = 1;
  end
else
  train = 1:size(g,2);
  % restrict test samples on test.
  restrict = 0;
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
  if ~restrict
    nEventsInClass(cl)= length(find(g(cl,test)));
  else
    nEventsInClass(cl)= length(find(g(cl,train)));
  end
end

if min(nEventsInClass)<nDivisions,
  msg= ['number of fold greater than samples in smallest class\n' ...
        'switching to leave-one-out'];
  bbci_warning(msg, 'sample', mfilename);
  [divTr, divTe]= sample_leaveOneOut(g, [1 1 xTrials(3:end)]);
  return;  
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
    if ~restrict
      [divTr{ti}, divTe{ti}]= sampleDivision(g(:,test), nDivisions, nPick);
    else  
      [divTr{ti}, divTe{ti}]= sampleDivision(g(:,train), nDivisions, nPick);
    end
  end
end

if ~restrict
  testrest = setdiff(1:size(g,2),test);
else
  testrest = setdiff(1:size(g,2),train);
end

if ~isempty(testrest)
  if ~restrict
    divTe = changedivTe(divTe,test);
    divTr = changedivTe(divTr,test);
    divTr = adddivTr(divTr,testrest);
  else
    divTr = changedivTe(divTr,train);
    divTe = changedivTe(divTe,train);
    divTe = adddivTr(divTe,testrest);    
  end
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
