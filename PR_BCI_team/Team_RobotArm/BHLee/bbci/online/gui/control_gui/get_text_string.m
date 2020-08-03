function str = get_text_string(value);
% GET_TEXT_STRING ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% change value in a evaluable string which result is value
%
% usage:
%    str = get_text_string(value);
%
% input:
%    value    the variable
% 
% output:
%    str      the string which evals to variable
%
% Guido Dornhege
% $Id: get_text_string.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


if ndims(value)>2
  error('not implemented so far');
end

s = size(value);
% do it for chars
if ischar(value)
  if isempty(value)
    str = '''''';
  elseif size(value,1)==1
    str = ['''' value ''''];
  else
    str = '[';
    for i = 1:size(value,1)
      str = sprintf('%s''%s'';',str,value(i,:));
    end
    str(end) = ']';
  end
  return;
end

% do it for empty areas
if prod(s)==0
  if iscell(value)
    str = '{}';
  else
    str = '[]';
  end
elseif prod(s)==1
  % no for single dimension values
  if iscell(value)
    str = sprintf('{%s}',get_text_string(value{:}));
  elseif ischar(value)
    str = sprintf('''%s''',value);
  elseif isnumeric(value) | islogical(value)
    str = sprintf('%g',value);
  elseif isstruct(value)
    str = sprintf('struct(');
    fi = fieldnames(value);
    for i = 1:length(fi)
      str = sprintf('%s,%s,%s',str,fi{i},get_text_string(getfield(value,fi{i})));
    end
    str = sprintf('%s)',str);
  else
    error('type not supported');
  end
else
  % do it iteratively for multi-dimensional data
  if iscell(value)
    str = '{';
  else
    str = '[';
  end
  
  for i = 1:s(1);
    for j = 1:s(2);
      if iscell(value)
        str = sprintf('%s%s,',str,get_text_string(value{i,j}));
      else
        str = sprintf('%s%s,',str,get_text_string(value(i,j)));
      end
    end
    str(end) = ';';
  end
  if iscell(value);
    str(end) = '}';
  else
    str(end) = ']';
  end
end
