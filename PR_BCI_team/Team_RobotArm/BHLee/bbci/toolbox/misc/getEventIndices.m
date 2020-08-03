function evInd= getEventIndices(mrk, varargin)
%clInd= getClassIndices(mrk, classes)
%clInd= getClassIndices(mrk, class1, ...)
%
% this function returns in indices of events of given classes
% in a marker structure
%
% IN   mrk       - marker structure
%      classes   - cell array of classes names or
%                  vector of class indices
%      classX    - class name or class index
%
% OUT  evInd     - event indices
%
% class names may include the wildcard '*' as first and/or last
% symbol.

ic= getClassIndices(mrk, varargin{:});
[dum,evInd]= find(mrk.y(ic,:));

evInd = unique(evInd);