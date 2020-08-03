function var = fromString(str)

if length(str)>=2 & strcmp(str(1:2),'{{')
  c = strfind(str,'{');
  c = find(diff(c)>1);c=c(1)-1;
  c1 = strfind(str,repmat('{',[1,c]));c1(1)=[];
  c2 = strfind(str,repmat('}',[1,c]));c2(end)=[];
  if length(c1)~=length(c2)
    error('???');
  end
  var = fromString(str(c1(1):c2(1)+c-1));
  n = ndims(var);
  for i = 2:length(c1)
    var = cat(n+1,var,fromString(str(c1(i):c2(i)+c-1)));
  end
elseif length(str)>=2 & strcmp(str(1:2),'[[')
  c = strfind(str,'[');
  c = find(diff(c)>1);c=c(1)-1;
  c1 = strfind(str,repmat('[',[1,c]));c1(1)=[];
  c2 = strfind(str,repmat(']',[1,c]));c2(end)=[];
  if length(c1)~=length(c2)
    error('???');
  end
  var = fromString(str(c1(1):c2(1)+c-1));
  n = ndims(var);
  for i = 2:length(c1)
    var = cat(n+1,var,fromString(str(c1(i):c2(i)+c-1)));
  end
else
  if str(1)==''''
    c = find(str=='''');
    c = c(2:end-1);
    for i = c(end:-1:1)
      str = [str(1:i-1),'''''',str(i+1:end)];
    end
  end
  var = eval(str);
end
