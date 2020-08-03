function [str, fid]= readBlockFromTape(fileName, bn)
%str= readBlockFromTape(fid/tapeName, <blockNumber>)
%str= readBlockFromTape(fid/tapeName, 'next')
%
% if no blockNumber is given, the number of blocks on the tape is returned.

if ischar(fileName),
  fileName= ['tape_' fileName];
  if ~ismember(fileName, '.'),
    fileName= [fileName '.m'];
  end
  fid= fopen(fileName, 'r');
  if fid==-1, error(sprintf('%s not found', fileName)); end
else
  fid= fileName;
end

if nargin<2,
  frewind(fid);
  str= 0;
  while ~isempty(readBlockFromTape(fid, 'next')),
    str= str+1;
  end
  frewind(fid);
  if nargout<2 & ischar(fileName),
    fclose(fid);
  end
  return;
else
  if ~isequal(bn, 'next'),
    frewind(fid);
    for bi= 1:bn-1;
      dummy= readBlockFromTape(fid, 'next');
    end
  end
end

str= {};
stopit= 0;
next_line_cont= 0;
while ~stopit,
  s= fgets(fid);
  if s==-1,
    stopit= 1;
  else
    s= fliplr(deblank(fliplr(deblank(s))));
    line_cont= next_line_cont;
    if length(s)>=3 & s(end-2:end)=='...',
      s= s(1:end-3);
      next_line_cont= 1;
    end
    stopit= feof(fid) | (~isempty(str) & isempty(s));
    if ~isempty(s),
      if line_cont,
        str{end}= cat(2, str{end}, s);
      else
        str= cat(2, str, {s});
      end
    end
  end
end
if ischar(fileName),
  fclose(fid);
end
