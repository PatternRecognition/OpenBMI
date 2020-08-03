function epo = proc_mergeClasses(epo,varargin)
%PROC_MERGECLASSES merges multiple classes to a single class.
%
% usage:
%      epo = proc_mergeClasses(epo,classes,...,<OPT>);
%
% Input:
% EPO         - epo structure
% CLASSES     - is a cell array of class names (out of epo.className)
%                 or a vector of class numbers. 
%
% OPT - struct or property/value list of optional properties:
% 'preserve' -  if 1, old classes are preserved and merged classes are
%               attached to the end of the class definitions (default 0)
%
% output:
%      epo       a epo structure with combined classes
%
%
%
% Example: if your epo comprises the classes 'left', 'right', 'foot' then
% proc_mergeClasses(epo,{'left' 'right'}) will give a two-class epo with
% the classes 'left & right' and 'foot'. An alternative way to write this
% is proc_mergeClasses(epo,1:2)
%
%
% See also: mrk_mergeClasses, proc_combineClasses
%
% Matthias Treder Aug 2010


merge = {};
opt = [];

% Parse inputs
for ii=1:numel(varargin)
  if iscell(varargin{ii})  % If class names are provided, turn them into indices
    merge{ii} = find(ismember(epo.className,varargin{ii}));
  elseif ischar(varargin{ii})  % belongs to opt
    opt= propertylist2struct(varargin{ii:end});
    break
  elseif isvector(varargin{ii})
    merge{ii} = varargin{ii};
  end
end
opt = set_defaults(opt, ...
      'preserve',0);


%% Merge classes
for ii=1:numel(merge)
  epo.y(end+1,:) = (sum(epo.y(merge{ii},:),1))>0;
  epo.className{end+1} = ...
    [sprintf('%s & ',epo.className{merge{ii}(1:end-1)}) epo.className{merge{ii}(end)}];
end


if opt.preserve==0
  % Delete old classes
  delC = unique([merge{:}]);
  epo.y(delC,:) = [];
  epo.className(delC) = [];
end
