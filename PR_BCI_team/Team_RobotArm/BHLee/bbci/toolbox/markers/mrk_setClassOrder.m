function mrk = mrk_setClassOrder(mrk,classnames,varargin);
%setClassOrder set the order of classes in mrk.className and mrk.y
%given by classnames
%
% usage: 
%   mrk = setClassOrder(mrk,classnames);
%
% input: 
%   mrk        - a usual mrk structure
%   classnames - a cell array (or a series of inputs) with
%                classnames or am integer array regarding ordering
%                in mrk.className (can also be an integer series)
%
% output:
%   mrk        - a new mrk structure with changed .className, .y
%                and maybe different number of classes
%
% Guido Dornhege, 19/09/2003
%
%bb: mrk_selectClasses does the same job. use that.

if ischar(classnames)
  classnames = {classnames};
end

if iscell(classnames)
  classnames = {classnames{:},varargin{:}};
end

if isnumeric(classnames)
  classnames = [classnames,varargin{:}];
end


if iscell(classnames)
  num = zeros(1,length(classnames));
  for i = 1:length(classnames)
    c = find(strcmp(mrk.className,classnames{i}));
    if isempty(c)
      mrk.y = cat(1,mrk.y,zeros(1,size(mrk.y,2)));
      mrk.className = cat(2,mrk.className,{classnames{i}});
      num(i)=length(mrk.className);
    else
      num(i) = c;
    end
  end
  classnames = num;
end

  
mrk.y = mrk.y(num,:);
mrk.className = {mrk.className{num}};

  
