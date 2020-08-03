function lines= str2lines(str)
%STR2LINES - Convert a string into a cell array of lines
%
%Description:
%  This functions converts a char array containing line feeds into
%  a cell array of char arrays without line feeds, i.e., each cell
%  hold one line of the original string.
%
%Usage:
%  LINES= str2lines(STR)
%
%Example:
%  c= str2lines(sprintf('blah\nblub')); c{1}, c{2}

iLF= find(str==10);
if isempty(iLF),
  lines= {str};
else
  lines= cell(1, length(iLF));
  iLF= [0 iLF];
  if iLF(end)<length(str),
    iLF= [iLF length(str)+1];
  end
  for ii= 1:length(iLF)-1,
    lines{ii}= str(iLF(ii)+1:iLF(ii+1)-1);
  end
end
