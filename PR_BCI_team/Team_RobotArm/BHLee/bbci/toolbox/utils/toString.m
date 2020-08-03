function str = toString(var, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'numeric_format', '%f', ...
                  'max_dim', inf, ...
                  'max_numel', inf);

if numel(var)>opt.max_numel,
  str= sprintf('[%s %s]', vec2str(size(var), '%d', 'x'), upper(class(var)));
  return;
end

str= [];
if islogical(var)
  if prod(size(var))==1
    if var==0, str = 'false'; else str = 'true';end
  elseif isempty(var)
    str = '[]';
  elseif ndims(var)<=2 && size(var,2)==1
    str = sprintf('%s''',toString(var', opt));
  elseif (ndims(var)<=2 && size(var,1)==1) || ...
        (ndims(var)<=2 && opt.max_dim>=2),
    str = '[';
    for i = 1:size(var,1)
      for j = 1:size(var,2)
        str = [str , toString(var(i,j), opt),','];
      end
      str = [str(1:end-1),';'];
    end
    str = [str(1:end-1),']'];
  elseif ndims(var)<=opt.max_dim,
    str = '[';
    nd = ndims(var);
    sv = size(var); sv(end)=[];
    va = permute(var,[nd,1:nd-1]);
    for i = 1:size(var,nd)
      v = va(i,:)';
      v = reshape(v,sv);
      str = [str toString(v, opt) ';'];
    end
    str = [str(1:end-1),']'];
  end
elseif isnumeric(var)
  if prod(size(var))==1
    if var==round(var),
      str = sprintf('%d',var);
    else
      str = sprintf(opt.numeric_format, var);
    end
  elseif isempty(var)
    str = '[]';
  elseif ndims(var)<=2 && size(var,2)==1
    str = sprintf('%s''',toString(var', opt));
  elseif (ndims(var)<=2 && size(var,1)==1) || ...
        (ndims(var)<=2 && opt.max_dim>=2),
    str = '[';
    for i = 1:size(var,1)
      for j = 1:size(var,2)
        str = [str , toString(var(i,j), opt),','];
      end
      str = [str(1:end-1),';'];
    end
    str = [str(1:end-1),']'];
  elseif ndims(var)<=opt.max_dim,
    str = '[';
    nd = ndims(var);
    sv = size(var); sv(end)=[];
    va = permute(var,[nd,1:nd-1]);
    for i = 1:size(var,nd)
      v = va(i,:)';
      v = reshape(v,sv);
      str = [str toString(v, opt) ';'];
    end
    str = [str(1:end-1),']'];
  end
elseif ischar(var)
  if ndims(var)<=2 & size(var,1)==1
    str = ['''',var,''''];
  elseif isempty(var)
    str = '''''';
  elseif ndims(var)<=2
    str = '[';
    for i = 1:size(var,1)
      str = [str toString(var(i,:), opt), ';'];
    end
    str = [str(1:end-1),']'];
  else
    str = '[';
    nd = ndims(var);
    sv = size(var); sv(end)=[];
    va = permute(var,[nd,1:nd-1]);
    for i = 1:size(var,nd)
      v = va(i,:)';
      v = reshape(v,sv);
      str = [str toString(v, opt) ';'];
    end
    str = [str(1:end-1),']'];
  end
elseif iscell(var)
   if prod(size(var))==1
    str = ['{',toString(var{1}, opt),'}'];
  elseif isempty(var)
    str = '{}';
  elseif ndims(var)<=2 & size(var,2)==1
    str = sprintf('%s''',toString(var', opt));
  elseif (ndims(var)<=2 && size(var,1)==1) || ...
        (ndims(var)<=2 && opt.max_dim>=2),
    str = '{';
    for i = 1:size(var,1)
      for j = 1:size(var,2)
        str = [str , toString(var{i,j}, opt),','];
      end
      str = [str(1:end-1),';'];
    end
    str = [str(1:end-1),'}'];
  elseif ndims(var)<=opt.max_dim,
    str = '{';
    nd = ndims(var);
    sv = size(var); sv(end)=[];
    va = permute(var,[nd,1:nd-1]);
    for i = 1:size(var,nd)
      v = va(i,:)';
      v = reshape(v,sv);
      str = [str toString(v, opt) ';'];
    end
    str = [str(1:end-1),'}'];
  end
elseif isstruct(var)
  if prod(size(var)) == 1
    str = 'struct(';
    a = fieldnames(var);
    for i = 1:length(a)
      str = [str, '''', a{i}, ''',', toString(getfield(var,a{i}), opt), ...
	     ','];
    end
    str = [str(1:end-~isempty(a)),')'];
  elseif isempty(var)
    str = 'struct([])';
  elseif ndims(var)<=2 && size(var,2)==1
    str = sprintf('%s''',toString(var', opt));
  elseif (ndims(var)<=2 && size(var,1)==1) || ...
        (ndims(var)<=2 && opt.max_dim>=2),
    str = '[';
    for i = 1:size(var,1)
      for j = 1:size(var,2)
        str = [str , toString(var(i,j), opt),','];
      end
      str = [str(1:end-1),';'];
    end
    str = [str(1:end-1),']'];
  elseif ndims(var)<=opt.max_dim,
    str = '[';
    nd = ndims(var);
    sv = size(var); sv(end)=[];
    va = permute(var,[nd,1:nd-1]);
    for i = 1:size(var,nd)
      v = va(i,:)';
      v = reshape(v,sv);
      str = [str toString(v, opt) ';'];
    end
    str = [str(1:end-1),']'];
  end
elseif isa(var, 'function_handle'),
  str= ['@' func2str(var)];
end

if isempty(str),
  str= sprintf('[%s %s]', vec2str(size(var), '%d', 'x'), upper(class(var)));
end
