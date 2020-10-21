function [ok, msg]= misc_checkType(variable, typeDefinition, propname, toplevel)
%MISC_CHECKTYPE - Check the type of a variable
%
%Synopsis:
%  misc_checkType(VARIABLE, TYPEDEF)
%  misc_checkType(EXPRESSION, TYPEDEF, VARNAME)
%
%Arguments:
%  VARIABLE:   Variable that is to be checked
%  TYPEDEF:    CHAR - Specification of type, see below.
%  EXPRESSION: Input that has 'no name' (-> it's value is checked)
%
%Remark: The second variant has to be used, if the 'variable' that is to
%  be checked is a field of a struct like cnt.clab.
%
%Returns:
%  nothing (throws an error in case of violation)
%
%Description:
%  This function checks that VARIABLE complies with the type specification
%  given in TYPEDEF. The following type specifications are implemented:
%    'DOUBLE' - value has to be a numeric array (of any size)
%    'DOUBLE[x]' with x being a nonnegative integer - value has
%          to be a numeric vector of length x. Here, row and column vectors
%          are both allowed. To force either row or column vectors of 
%          length x, use 'DOUBLE[x -]' resp. 'DOUBLE[- x]', see below.
%    'DOUBLE[x-y]' with x,y being a nonnegative integers - value has
%          to be a numeric vector with a length between x and y.
%    'DOUBLE[x y z]' with x,y,z begin nonnegative intergers  -
%          value has to be a matrix of numeric values of size [x y z].
%          Each dimension specifier can also be a range:
%            'a-b' length of that dimension has to be between a and b,
%            'a-'  length has to be great than or equal to a,
%            '-b'  length has to be less than or equal to b,
%            '-'   length can be anything
%    'INT' - value has to be a numeric array (class double, not int16) with
%             all elements being integers
%            of any size. Size can be specified as above for 'DOUBLE'.
%    'BOOL' - value has to be a logical array or a numeric array with all
%            elements being 0 or 1. Size can be specified as above.
%    'CHAR' - value has to be a character array. Size can be specified as
%            above for 'DOUBLE'.
%    'CHAR(val1 val2)' - value has to be a character array and it has to be
%            equal to one of the specified strings val1 val2 (an arbitrary 
%            number of values can be specified).
%    'FUNC' - value has to be a function handle
%    'GRAPHICS' - value has to be a graphics or Java handle
%    'STRUCT' - value has to be a struct array
%    'STRUCT(fld1 fld2)' - values has to be a struct array and the specified
%            fields 'fld1' and 'fld2' must exist (an arbitrary number of fields
%            can be specified).
%    'CELL' - value has to be a cell array
%    'CELL{<TYPE>}' - value has to be a cell array and the contents of each 
%            cell have to be of type <TYPE> (example 'CELL{CHAR}').
%    'PROPLIST' - Property/value list
%    'PROPSPEC' - Property specification as used in opt_setDefaults
%    Furthermore, alternative type specifications can be combined with the
%    operator '|' (example 'CHAR|DOUBLE[3]' for a color specification).
%    By default, the empty value is always accepted. In order to force
%    nonempty values, prepend '!' to the type specification (such as
%    '!CHAR').
%
%  Type checking can be switched off (and on again) by the function
%  bbci_typechecking which is useful for time critical operations.
%
%See also opt_checkTypes, bbci_typechecking.
%
%Examples:
%  vec= 1:5;
%  misc_checkType(vec, 'DOUBLE')
%  misc_checkType(1:5, 'DOUBLE', 'vec')  % alternative use (mainly internal)
%  misc_checkType(vec, 'DOUBLE[4]')
%  misc_checkType(vec, 'DOUBLE[- 5]')
%  misc_checkType(vec, 'DOUBLE[- 5]')
%
%  colormap= hot(21);
%  misc_checkType(colormap, 'DOUBLE[- 3]')
%  colormap= rand(21, 4);
%  misc_checkType(colormap, 'DOUBLE[- 3]')
%
%  linecolor= 'green';   % allowed would be 'g' or [0 1 0]
%  misc_checkType(linecolor, 'CHAR[1]|DOUBLE[3]')
%
%  cnt= struct('x',randn(1000,2), 'clab',{'C3','C4'});
%  misc_checkType(cnt, 'STRUCT(x fs clab)')
%
%  clab= str_cprintf('C%d', 1:6)
%  misc_checkType(clab, 'CELL{CHAR}')
%  clab{end}= 3.14;
%  misc_checkType(clab, 'CELL{CHAR}')
%
%  cnt.clab= str_cprintf('C%d', 1:6);
%  misc_checkType(cnt.clab, 'STRUCT(desc)', 'cnt.clab');

% 06-2012 Benjamin Blankertz


global BTB

% if ~BTB.TypeChecking, return; end

if nargin<4,
  toplevel= 1;
end
if nargin<3,
  propname= inputname(1);
  if isempty(propname),
    error('First argument must be a variable, or a third argument must be provided.');
  end
end

ok= [];
msg= '';
if isempty(typeDefinition),
  ok= 1;
elseif any('|'==typeDefinition),
  ii= min(find(typeDefinition=='|'));
  ok= or(misc_checkType(variable, typeDefinition(1:ii-1), propname, 0), ...
      misc_checkType(variable, typeDefinition(ii+1:end), propname, 0));
else
  allow_empty= 1;
  if typeDefinition(1)=='!',
    allow_empty= 0;
    typeDefinition(1)= [];
  end
end

if ~isempty(ok),
  % type check was already performed
elseif isempty(variable) && allow_empty,
  ok= 1;
elseif isempty(variable) && ~allow_empty,
  ok= 0;
  msg= sprintf(['Type error: variable ''%s'' must not be empty ' ... 
                '(expected type is !%s)'], ...
                 propname, typeDefinition);
elseif str_matchesHead('DOUBLE', typeDefinition),
  ok= isnumeric(variable);
  if ok,
    [ok, msg]= size_check(variable, typeDefinition(length('DOUBLE')+1:end), ...
                          propname);
  end
elseif str_matchesHead('INT', typeDefinition),
  if isnumeric(variable),
    ok= all(variable(:)==floor(variable(:)));
  else
    ok= 0;
  end
  if ok,
    [ok, msg]= size_check(variable, typeDefinition(length('INT')+1:end), ...
                          propname);
  end
elseif str_matchesHead('BOOL', typeDefinition),
  ok= islogical(variable) || isequal(variable, 0) || isequal(variable, 1);
  if ok,
    [ok, msg]= size_check(variable, typeDefinition(length('BOOL')+1:end), ...
                          propname);
  end
elseif str_matchesHead('CHAR', typeDefinition),
  ok= ischar(variable);
  spec= typeDefinition(length('CHAR')+1:end);
  if ok && ~isempty(spec),
    if strcmp(spec([1 end]), '()'),
      allowedValues = textscan(spec(2:end-1), '%s');
      if ~any(strcmpi(variable,allowedValues{1}));        
        ok = 0;
        msg= sprintf('Invalid value ''%s'' of variable ''%s''. Allowed are only the strings: %s', ...
                     variable, propname, spec(2:end-1));
      end
    else
        [ok, msg]= size_check(variable, typeDefinition(length('CHAR')+1:end), ...
                              propname);
    end
  end
elseif str_matchesHead('FUNC', typeDefinition),
  ok= isa(variable, 'function_handle');
elseif str_matchesHead('GRAPHICS', typeDefinition),
  ok= ishandle(variable);
elseif str_matchesHead('CELL', typeDefinition),
  ok= iscell(variable);
  spec= typeDefinition(length('CELL')+1:end);
  if ok && ~isempty(spec),
    if ~strcmp(spec([1 end]), '{}'),
      ok= 0;
      msg= sprintf('Invalid specification ''%s'' of CELL type in variable ''%s''', ...
            spec, propname);
    end
    vn= sprintf('%s (within cells)', propname);
    [ok_array, msg_array]= ...
        cellfun(@misc_checkType, variable, ...
                repmat({spec(2:end-1)}, size(variable)), ...
                repmat({vn}, size(variable)), ...
                repmat({0}, size(variable)), ...
                'UniformOutput', 0);
    ok= all([ok_array{:}]);
    if ~ok,
      msg= msg_array{min(find(~[ok_array{:}]))};
    end
  end
elseif str_matchesHead('STRUCT', typeDefinition),
  ok= isstruct(variable);
  spec= typeDefinition(length('STRUCT')+1:end);
  if ok && ~isempty(spec),
    if ~strcmp(spec([1 end]), '()'),
      ok= 0;
      msg= sprintf('Invalid specification ''%s'' of STRUCT type in variable ''%s''', ...
                   spec, variable);
    end
    requiredFields= textscan(spec(2:end-1), '%s');
    ok= isfield(variable, requiredFields{1});
    if ~all(ok),
      msg= sprintf('Missing obligatory field(s) in variable ''%s'': %s', ...
                   propname, str_vec2str(requiredFields{1}(~ok)));
      ok= 0;
    end
  end
elseif str_matchesHead('PROPLIST', typeDefinition),
  if isempty(variable) || ...
        ( iscell(variable) && length(variable)==1 && isstruct(variable{1}) ),
    ok= 1;
  elseif iscell(variable) && ndims(variable)==2 && size(variable,1)==1 && ...
          mod(length(variable),2)==0,
    ok= all(cellfun(@ischar, variable(1:2:end)));
  else
    ok= 0;
  end
elseif str_matchesHead('PROPSPEC', typeDefinition),
  if isempty(variable) || ...
        ( iscell(variable) && ndims(variable)==2 && ...
          (size(variable,2)==2 || size(variable,2)==3) ),
    ok= all(cellfun(@ischar, variable(:,1)));
    if size(variable,2)==3,
      ok= ok && all(cellfun(@ischar, variable(:,3)));
    end
  else
    ok= 0;
  end
else
  ok= 0;
  msg= sprintf('Unknown type: %s in variable ''%s''', typeDefinition, propname);
end

if ~ok
  if isempty(msg),
    msg= sprintf('Type error in variable ''%s'': expected type is %s', ...
                 propname, typeDefinition);
  end
  if toplevel,
    error(msg);
  end
end
if nargout==0,
  clear ok
end




function [ok, msg]= size_check(variable, sizeDefinition, propname)

msg= '';
if isempty(sizeDefinition),
  ok= 1;
  return;
end

if ~strcmp(sizeDefinition([1 end]), '[]'),
  ok= 0;
  msg= sprintf('Invalid size specification ''%s'' for variable ''%s''', ...
        sizeDefinition, propname);
end

% Parse the string that defines the size
k= 0;
[def, remain]= strtok(sizeDefinition(2:end-1));
while ~isempty(def),
  k= k+1;
  idash= find(def=='-');
  if isempty(idash),
    if isempty(def),
      sizerange(k,:)= [0 inf];
    else
      sizerange(k,:)= str2double(def) * [1 1];
    end
  else
    if length(def)==1,  % def=='-'
      sizerange(k,:)= [0 inf];
    elseif idash==1,
      sizerange(k,:)= [0 str2double(def(idash+1:end))];
    elseif idash==length(def),
      sizerange(k,:)= [str2double(def(1:idash-1)) inf];
    else
      sizerange(k,:)= str2double({def(1:idash-1), def(idash+1:end)});
    end
  end
  [def, remain]= strtok(remain);
end

% Check that the variable has the specified size
nsz= k;
if nsz==0,
  ok= 1;
elseif nsz==1,
  % If only one dimension is specified, we allow row as well as colomn vectors
  ok= ndims(variable)==2 && ...
      any(size(variable)==1) && ...
      numel(variable)>=sizerange(nsz,1) && numel(variable)<=sizerange(nsz,2);
else
  if nsz~=ndims(variable),
    ok= 0;
    msg= sprintf('Mismatch in #dim of variable ''%s'': expected %d but got %d', ...
                 propname, nsz, ndims(variable));
    return;
  end
  ok= 1;
  for k= 1:nsz,
    ok= ok && size(variable,k)>=sizerange(k,1) && ...
        size(variable,k)<=sizerange(k,2);
  end
end

if ~ok,
  receivedSize= size(variable);
  if length(receivedSize)==2 && any(receivedSize==1) && nsz==1,
    receivedSize= length(variable);
  end
  msg= sprintf('Size mismatch for variable ''%s'': expected %s but got [%s]', ...
        propname, sizeDefinition, str_vec2str(receivedSize,'%d',' '));
end