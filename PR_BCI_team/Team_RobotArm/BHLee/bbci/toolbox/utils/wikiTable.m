function wikiTable(data, filename, varargin)
% wikiTable - Generate WIKI table from matrix
%
% Synopsis:
%   wikiTable(data)
%   wikiTable(data,filename, 'Property', 'Value', ...)
%   
% Arguments:
%  data: [n m] matrix. The data to output as a table.
%  filename: String or file handle. If string is given, open this file
%      and write. If file handle, write to this file. Use filename==1 to
%      write to standard output. Default value: 1
%   
% Properties:
%  title: String to put into the upper left corner of the table. Default
%      value: ''
%  col_title: String or cell string. Label for all or individual
%      columns. Default value: ''.
%  row_title: String or cell string. Label for all or individual
%      rows. Default value: ''.
%  col_format: String or cell string. Format string for the data entries of the
%      table. If cell string, use separate format for each column of the
%      data. Default value: '%4.2f' 
%  separator: String. Separator for table entries. Default value: '||'
%   
%   
% Examples:
%   wikiTable([1 2;3 4])
%     || 1.00 || 2.00 ||
%     || 3.00 || 4.00 ||
%   wikiTable([1 2;3 4], 1, 'col_format', {'%4.2f', '%i'})
%     || 1.00 || 2 ||
%     || 3.00 || 4 ||
%   wikiTable([1 2;3 4], 1, 'col_format', '%i', 'title', 'T', 'col_title', {'C1', 'C2'})
%     || T || C1 || C2 ||
%     ||   || 1 || 2 ||
%     ||   || 3 || 4 ||
%
% See also: fprintf,fopen,latexTable
% 

% Author(s), Copyright: Anton Schwaighofer, May 2005
% $Id: wikiTable.m,v 1.2 2005/05/26 13:03:55 neuro_toolbox Exp $

error(nargchk(1, inf, nargin));
if nargin<2,
  filename = [];
end
[nRows, nCols] = size(data);

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'title', '', ...
                  'col_title', [], ...
                  'row_title', [], ...
                  'col_format', '%4.2f', ...
                  'separator', '||');

% Make sure that all options can be passed as strings: Expand to required
% size if necessary
if ischar(opt.row_title),
  opt.row_title= repmat({opt.row_title}, 1, nRows);
end
if ischar(opt.col_format),
  opt.col_format= repmat({opt.col_format}, 1, nCols);
end
if isempty(opt.row_title),
  % No row title given, but we have a title to be put into the upper left
  % corner: Preced each line with an empty row title
  if ~isempty(opt.title),
    opt.row_title= repmat({''}, 1, nRows);
  end
end

% Extract maximum length of title and row titles, so that the table is
% (mostly) nicely arranged even in plaintext 
maxLen = -Inf;
if ~isempty(opt.row_title),
  if ischar(opt.row_title),
    maxLen = length(opt.row_title);
  else
    maxLen = max(cell2mat(apply_cellwise(opt.row_title, 'length')));
  end
end
if ~isempty(opt.title),
  maxLen = max(maxLen, length(opt.title));
end

% If a file name is given, open that file for reading
if ischar(filename) & ~isempty(filename),
  file = fopen(filename, 'wt');
  if file<0,
    error(sprintf('Unable to open file ''%s'' for writing', filename));
  end
elseif isempty(filename),
  % If no filename given, write to standard output
  file = 1;
else  
  % Assume that the argument is a valid file handle
  file = filename;
end

% Write first header line, if either title or column titles are given
if ~isempty(opt.title) | ~isempty(opt.col_title),
  % Either of the two might be empty or given as a single string: Expand
  if isempty(opt.col_title),
    opt.col_title = '';
  end
  if ischar(opt.col_title),
    opt.col_title= repmat({opt.col_title}, 1, nCols);
  end
  if ~isempty(opt.title) | ~isempty(opt.row_title),
    fprintf(file, ['%s %' num2str(maxLen) 's '], opt.separator, opt.title);
  end
  % Create a cell array with separators and title strings, fprintf will
  % process it sequentially (sep, title, sep, title, ...)
  c = vertcat(repmat({opt.separator}, [1 nCols]), opt.col_title(:)');
  fprintf(file, '%s %s ', c{:});
  fprintf(file, '%s\n', opt.separator);
end

% Print out each row with eventual row title
for i = 1:nRows,
  if ~isempty(opt.row_title),
    if ischar(opt.row_title),
      t = opt.row_title;
    else
      t = opt.row_title{i};
    end
    fprintf(file, ['%s %' num2str(maxLen) 's '], opt.separator, t);
  end
  formatstr = sprintf('%%s %s ', opt.col_format{:});
  % Cell with separators and data, process it sequentially (as above)
  c = vertcat(repmat({opt.separator}, [1 nCols]), num2cell(data(i,:)));
  fprintf(file, formatstr, c{:});
  fprintf(file, '%s\n', opt.separator);
end

if ischar(filename),
  status = fclose(file);
  if status<0,
    error(sprintf('Unable to close file ''%s''', filename));
  end
end

