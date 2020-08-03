function classIdcs= getEquiSubset(equi)
%classIdcs= getEquiSubset(equi)

subs= equiSubset(equi.idcs);
if ~isfield(equi, 'classes'),
  half= floor(length(subs)/2);
  equi.classes= {1:half, half+1:length(subs)};
end

nClasses= length(equi.classes);
classIdcs= cell(1, nClasses);
for ci= 1:nClasses,
  classIdcs{ci}= [subs{equi.classes{ci}}];
end
