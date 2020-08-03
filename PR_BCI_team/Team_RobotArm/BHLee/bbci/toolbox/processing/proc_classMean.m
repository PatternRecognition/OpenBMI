function epo= proc_classMean(epo, clInd)
%epo= proc_classMean(epo, <clInd>)
%
% calculate the mean of all events belonging to the classes with indices 
% in clInd. if clInd is not given, the mean of all events is calculated.
%
% SEE proc_average

if ~exist('clInd','var'), 
  evInd= {1:size(epo.x, 3)};
  epo.className= {'mean'};
else
  evInd= cell(1, length(clInd));
  for ic= 1:length(clInd),
    evInd{ic}= find(epo.y(clInd(ic),:));
  end
  epo.className= {epo.className{clInd}};
end

nClasses= length(evInd);
epo.N= zeros(1, nClasses);
xm= zeros(size(epo.x,1), size(epo.x,2), nClasses);
for ic= 1:nClasses,
  xm(:,:,ic)= mean(epo.x(:,:,evInd{ic}), 3);
  epo.N(ic)= length(evInd{ic});
end
epo.x= xm;
epo.y= eye(nClasses);
