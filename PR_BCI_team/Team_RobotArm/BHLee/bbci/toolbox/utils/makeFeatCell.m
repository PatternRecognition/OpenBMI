function data = makeFeatCell(dat)
%
% DUDU
% 03.07.02

if ~iscell(dat) & ~isstruct(dat)
    data = {dat};
end

if isstruct(dat)
  data = makeFeatCell(dat.x);
end

if iscell(dat)
  data = {};
  for i =1:length(dat)
    data = {data{:}, concat(dat{i})};
  end
end

 
