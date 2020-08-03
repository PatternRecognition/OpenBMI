function ismatch= strpatterncmp(pat, str)
%ismatch= strpatterncmp(pat, str)
%
% IN  pat     - pattern (string)
%               which may include the wildcard '*' at the beginning
%               and/or the end, i.e., only patterns like '*blah', 
%               'blah*' or '*blah*' implemented, 
%     str     - string
%
% OUT ismatch - 1 if the pattern matches the string, 0 otherwise

% bb ida.first.fhg.de, 10/2003

if isempty(pat),
  ismatch= isempty(str);
  return;
end

if pat(1)=='*' & pat(end)=='*',
  substr= pat(2:end-1);
  ls= length(substr);
  ismatch= 0;
  k= 1;
  while ~ismatch & k<=length(str)-ls+1,
    ismatch= all(substr==str(k:k+ls-1));
    k= k+1;
  end
elseif pat(1)=='*',
  ismatch= strpatterncmp(fliplr(pat), fliplr(str));
elseif pat(end)=='*',
  pl= length(pat)-1;
  ismatch= strncmp(pat(1:pl), str, pl);
else
  nstars= sum(pat=='*');
  if nstars==1,
    is= find(pat=='*')-1;
    nr= length(pat)-is-1;
    ismatch= strncmp(pat(1:is), str, is) & ...
             strncmp(pat(is+2:end), str(end-nr+1:end), nr);
  elseif nstars>1,
    error('can only handle at most one wildcard');
  else
    ismatch= strcmp(pat, str);
  end
end
