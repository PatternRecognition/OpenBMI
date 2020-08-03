function fv= setGoal(fv, clInd, clNames)
%fv= setGoal(fv, clInd, <clNames>)
%ny= setGoal(y, clInd)

nClasses= length(clInd);
if isstruct(fv),
  nEvents= size(fv.x, ndims(fv.x));
  fv.y= zeros(nClasses, nEvents);
  for ic= 1:nClasses,
    fv.y(ic, clInd{ic})= 1;
  end
  if exist('clNames', 'var'),
    fv.className= clNames;
  end
else
  nEvents= size(fv, 2);
  y= zeros(nClasses, nEvents);
  for ic= 1:nClasses,
    y(ic, clInd{ic})= 1;
  end
  fv= y;
end
