function name = getShortClassname(NameCell)
% getShortClassname - Get short-hands (upper case of the first character) for class names
%
% Example:
%  getShortClassname({left, right}) -> 'LR'
%
% Ryota Tomioka, 2007


name = '';
for i=1:length(NameCell)
  name = [name, upper(NameCell{i}(1))];
end

