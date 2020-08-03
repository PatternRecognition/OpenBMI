function cpy= copyStruct(src, varargin)
%cpy= copyStruct(src, f1, ...)
%
% cpy is a copy of struct src, except for fields f1, ... 

cpy= [];
names= fieldnames(src);
for fi= 1:length(names),
  if isempty(strmatch(names{fi}, varargin)),
    cpy= setfield(cpy, names{fi}, getfield(src, names{fi}));
  end
end
