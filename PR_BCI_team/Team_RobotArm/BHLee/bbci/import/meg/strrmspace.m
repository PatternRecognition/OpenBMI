function cstr= strrmspace(cstr)
%head= strhead(str)
%
% returns the string up to (not including) the first space
% works also for cells of strings

if ~iscell(cstr), cstr={cstr}; end
for il= 1:length(cstr),
  cstr{il}(findstr(' ', cstr{il}))= [] ;
end
