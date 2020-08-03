function str= getBlockFromTape(fileName, bn)
%str= getBlockFromTape(fileName/fid, <blockNumber>)
%
% if blockNumber is 0, the number of blocks on the tape is returned.

if ischar(fileName),
  fileName= ['tape_' fileName '.m'];
  fid= fopen(fileName, 'r');
  if fid==-1, error(sprintf('%s not found', fileName)); end
else
  fid= fileName;
end

if nargin>1,
  frewind(fid);
  if bn==0,
    str= 0;
    while ~isempty(getBlockFromTape(fid)),
      str= str+1;
    end
    return;
  else
    for bi= 1:bn-1;
      dummy= getBlockFromTape(fid);
    end
  end
end

str= '';
stopit= 0;
while ~stopit,
  s= fgets(fid);
  if s==-1,
    stopit= 1;
  else
    s= fliplr(deblank(fliplr(deblank(s))));
    if length(s)>=3 & s(end-2:end)=='...',
      s= s(1:end-3);
    end
    stopit= isempty(s) | feof(fid);
    % append the current line to the lines so far and
    % add a newline (==char(10))
    str= [str s char(10)];
  end
end
str= str(1:end-1);
if ischar(fileName),
  fclose(fid);
end


