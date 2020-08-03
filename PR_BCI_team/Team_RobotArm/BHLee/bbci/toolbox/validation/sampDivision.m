function [divTr, divTe, nPick]= sampDivision(g, nTrain, nPick)
%[divTr, divTe]= sampleDivision(goal, nTrain, nPick)
%
%g= randn(1,100)>0;
%g(2,:)= 1-g;
%[divTr, divTe]= sampleDivision(g, 5);

nClasses= size(g,1);
nEventsInClass= zeros(1, nClasses);
for cl= 1:nClasses,
  nEventsInClass(cl)= length(find(g(cl,:)));
end
if ~exist('nPick','var'),
  nPick= nEventsInClass;
end

divTr= [];
divTe= [];

for cl= 1:nClasses,
  ci= find(g(cl,:));
  idx= randperm(nEventsInClass(cl));
  divTr = [divTr ci(idx(1:nTrain(cl)))];
  divTe = [divTe ci(idx(nTrain(cl)+1:nPick(cl)))];
end


divTr = divTr(randperm(length(divTr)));
divTe = divTe(randperm(length(divTe)));





