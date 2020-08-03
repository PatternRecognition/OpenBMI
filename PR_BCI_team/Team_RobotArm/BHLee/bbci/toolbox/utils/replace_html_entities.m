function str = replace_html_entities(str,direction)

% REPLACE_HTML_ENTITIES -  replaces special symbols such as '<' by their
%             according HTML entities for correct display in HTML or XML.
%             Can also perform the backward substitution (replacing HTML
%             entities by symbols).
%
%Usage:
% str = pyff(str,<direction>)
%
%IN:
% str - input string or cell array of strings
% direction - direction of substitution, 'forward' (default) replaces
%             symbols by HTML entities, 'backward' does the opposite 
%
%OUT:
% str - string with special symbols replaced by HTML entities
%
%EXAMPLE:
% replace_html_entities('If A < B then C')

% Matthias Treder 2010

if nargin<2, direction = 'forward'; end


% TO DO ---- REPLACE literal special letters by their ASCII code !!!

% Specify symbols (ie their ASCII code) and their corresponding entities in matched order
source = {'´'       '^'      'ä'      'Ä'      '<'    '>' ...
  'ö'      'Ö'     };
target = {'&acute;' '&circ;' '&auml;' '&Auml;' '&lt;' '&gt;' ...
  '&ouml;' '&Ouml;'};

% For backward substitution switch source and target
if strcmp(direction,'backward')
  dummy = source;
  source = target;
  target = dummy;
end

% Replace
if ischar(str)
  for ii=1:numel(source)
      str = strrep(str,source{ii},target{ii});
  end
elseif iscell(str)
  for jj=1:numel(str)
    str{jj} = replace_html_entities(str{jj});
  end
end