function [mrk, ev, clInd]= mrk_mergeClasses(mrk, varargin)
%MRK_MERGECLASSES - Merge Some Classes of a Marker Struct Into One
%
%Description:
% This function merges given classes into one new class. All other
% classes are kept. The name of the new joint class is the name of the 
% first class in the list. Alternatively, if the last class name does not
% exist yet it is assumed to be the new class name.
%
%Synopsis:
% [MRK, EV_IND, CL_IND]= mrk_mergeClasses(MRK, CLASSES)
% [MRK, EV_IND, CL_IND]= mrk_mergeClasses(mrk, CLASS1, ...)
%
%Arguments
% MRK: marker structure
% CLASSES: cell array of classes names or vector of class indices
% CLASSX: class name or class index
%
%Returns:
% MRK: structure containing the merged class
% EV_IND: indicies of all events of the new class
% CL_IND: indicies of all classes that have been merged.
%
%Example:
% mrk= mrk_mergeClasses(mrk, 'left', 'left no-click');

% bb 08/03, ida.first.fhg; mt 2011


clInd= getClassIndices(mrk, varargin{:});

cc = cell_flaten(varargin);
if ~any(ismember(mrk.className,cc{end}))
  newLabel = cc{end};
else
  newLabel = cc{1};
end


%% find events belonging to the specified classes ...
ev= find(any(mrk.y(clInd,:),1));

mrk.y(clInd(1),:)= ismember(1:size(mrk.y,2), ev);
mrk.y(clInd(2:end),:)= [];

mrk.className{clInd(1)}= newLabel;
mrk.className(clInd(2:end))= [];
