function [mrk, ev]= mrk_selectClasses(mrk, varargin)
%[mrk, ev]= mrk_selectClasses(mrk, classes)
%[mrk, ev]= mrk_selectClasses(mrk, class1, ...)
%
% this function selects events from a marker structure that belong
% to given classes. a typical application is to select from a multi-class
% experiment a two-class subproblem.
%
% IN   mrk     - marker structure
%      classes - cell array of classes names or
%                vector of class indices
%      classX  - class name or class index
%
% OUT  mrk     - structure containing only selected events
%      ev      - indices of select events
%
% EG   mrk_lr= mrk_selectClasses(mrk, {'left', 'right'});
%      mrk_12= mrk_selectClasses(mrk, [1 2]);
%      mrk_lr= mrk_selectClasses(mrk, 'left', 'right');
%
% class names may include the wildcard '*' as first and/or last
% symbol, see getClassIndices.

% bb 03/03, ida.first.fhg.de

if length(varargin)>=1 && ischar(varargin{1}) && strcmp(varargin{1},'remainclasses');
  remainclasses = 1;
  clInd= getClassIndices(mrk, varargin{2:end});
else
  remainclasses = 0;
  clInd= getClassIndices(mrk, varargin{1:end});
end  

%% the following is done to keep the order of the specified classes
mrk.y= mrk.y(clInd,:);
mrk.className= mrk.className(clInd);


%% select events belonging to the specified classes ...
ev= find(any(mrk.y, 1));
if remainclasses
  mrk= mrk_selectEvents(mrk, ev,'remainclasses',1);
else
  mrk= mrk_selectEvents(mrk, ev);
end

