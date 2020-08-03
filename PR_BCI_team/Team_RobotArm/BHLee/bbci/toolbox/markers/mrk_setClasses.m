function mrk= mrk_setClasses(mrk, idx, className)
%mrk= mrk_setClasses(mrk, idx, className)
%
% define new classes for a marker structure.
%
% IN  mrk       - marker structure
%     idx       - cell array of epoch indices, one cell for each class
%     className - cell array of class names
%
% OUT mrk       - updated marker structure

% bb 03/03, ida.first.fhg.de


nEpochs= length(mrk.pos);
if ~iscell(idx),
  idx= {idx};
end
nClasses= length(idx);
if exist('className', 'var'),
  if nClasses==1 && ~iscell(className),
    className= {className};
  end
  mrk.className= className;
else
  if length(mrk.className)~=nClasses,
    error('number of classes changed: provide class names');
  end
end

mrk.y= zeros(nClasses, nEpochs);
for ic= 1:nClasses,
  mrk.y(ic,:)= ismember(1:nEpochs, idx{ic});
end

mrk= mrk_selectEvents(mrk, [idx{:}]);
