function str = get_save_string(var,begin);
% str = get_save_string(var,varName)
%
% transform a variable with its name to an eval'able string
% which would set the variable 'varName' to the value 'var'.
% Attention: 1. only two-dimensional arrays are allowed.
%            2. logical values are converted to doubles.
%            3. some numerical inprecision due to truncating,
%               for reasons of legibility.
%
% IN: var     - the variable to store.
%     varName - its name.
% OUT:str     - a string containing matlab code.

% kraulem 07/05

str = '';

if isstruct(var)
  % struct; possibly with structs as fields.
  fi = fieldnames(var);
  for ii = 1:length(fi);
    sub_begin = [begin '.' fi{ii}];
    str = [str, get_save_string(getfield(var,fi{ii}),sub_begin)];
  end
elseif isnumeric(var)
  % double array
  if ndims(var)>2
    error('Variables must not have more than 2 dimensions');
  end
  % enter row by row into the array.
  siz = size(var);
  str = sprintf('%s = zeros(%i,%i);\n',begin,siz(1),siz(2));
  for ii = 1:siz(1)
    str = [str, sprintf('%s(%i,:) = [%s];\n',begin,ii,num2str(var(ii,:)))];
  end
elseif iscell(var)
  % cell array
  if ndims(var)>2
   error('Variables must not have more than 2 dimensions');
  end
  % enter row by row into the array.
  siz = size(var);
  str = sprintf('%s = cell(%i,%i);\n',begin,siz(1),siz(2));
  for ii = 1:siz(1)
    for jj = 1:siz(2)
      sub_begin = sprintf('%s{%i,%i}',begin,ii,jj);
      str = [str, get_save_string(var{ii,jj},sub_begin)];
    end
  end
elseif ischar(var)
  % String
  str = sprintf('%s = ''%s'';\n', begin, var);
elseif islogical(var)
  % Logical value. Convert to double.
  str = get_save_string(double(var),begin);
end








