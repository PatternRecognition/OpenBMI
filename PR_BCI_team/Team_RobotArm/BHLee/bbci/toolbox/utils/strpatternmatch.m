function ind= strpatternmatch(pats, strs)
%function ind= strpatternmatch(pats, strs)
%
% IN  pats - pattern (string) or patterns (cell array of strings)
%            which may include the wildcard '*' at the beginning
%            and/or the end, i.e., only patterns like '*blah', 
%            'blah*' or '*blah*' implemented, 
%
%     strs - cell array of strings (or just a string)
%
% OUT ind  - indices of those strings which are matched by the
%            (resp. by any of the) pattern(s).
%
% see strpatterncmp

% bb, 10/2003 ida.fhg.de


if ~iscell(pats), pats= {pats}; end
if ~iscell(strs), strs= {strs}; end

ind= [];
for ii= 1:length(strs),
  ismatch= 0;
  for jj= 1:length(pats),
    if strpatterncmp(pats{jj}, strs{ii}),
      ismatch= 1;
    end
  end
  if ismatch,
    ind= [ind, ii];
  end
end
