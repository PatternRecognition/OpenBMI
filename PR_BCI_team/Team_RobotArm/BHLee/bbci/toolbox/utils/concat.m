function data = concat(dat)
%
% DUDU
% 03.07.2002

if ~isstruct(dat) & ~iscell(dat)
  data = dat;
end

if isstruct(dat)
  data = concat(dat.x);
end

if iscell(dat)
  data = [];
  for i = 1:length(dat);
    data = cat(1,data,concat(dat{i}));
  end
end


