function latexTable(data, file, varargin);
% latexTable - Prettyprint a matrix as LaTeX tabular code
%
% Synopsis:
%   latexTable(data)
%   latexTable(data,file)
%   latexTable(data,file,'Property',Value,...)
%   
% Arguments:
%  data: [rows cols] matrix or cell. The data to be output
%  file: String or file handle. The LaTeX table will be written to this
%      file. Use file handle 1 to output to standard output. If no
%      extension is given, '.tex' is appended. Default: 1
%   
% Properties:
%  format_entry: String or cell array. Defines the print format for one
%      table entry. If given as string, use this for all entries of the
%      table. If cell array of size [1 cols], use entries for individual
%      columns. If cell array of size [1 cols+1], use the first entry for the
%      row header. Default: '%4.2f'
%  format_table: String. Format string that is put into LaTeX's tabular
%      command. A default is built based on data size and properties
%      'vlines', this will be something like '@{}l|r...r@{}'
%  col_title: String or [1 d] cell array. Column headers that are put in
%      the first row of the table. d must be a divisor of the number of
%      columns. If d<cols, each string is the header of d/cols
%      columns. Default: {}
%  row_title: String or [1 rows] cell array, is put into the first column
%      of the table. If string given, use the same value for each
%      row. Default: {}
%  title: String, is put in the upper left corner of the table in bold
%      face by default, or in the markup given by option <title_markup>
%  title_markup: String, markup command for the title string. Curly
%      braces and trailing backslash must not be included. Default: 'textbf' 
%  table_type: String. LaTeX command that is used to generate the table,
%      this will typically be one of {'tabular', 'longtable'}. Default:
%      'tabular'
%  col_summary: String, one of {'', 'mean', 'sum'}. If non-empty, an
%      extra row is output that contains the mean values (resp. sums) of
%      the corresponding columns
%  hlines: String, one of {'default', 'above', 'below', 'all', 'none'}
%      Controls whether horizontal lines are printed after the row of
%      column titles ('default'), above, below, or above and below the
%      whole table.
%  vlines: String, one of {'default', 'left', 'right', 'all', 'none'}
%      Controls whether vertical lines are printed left, right, left and
%      right of the table. This option affects the automatic generation
%      of format_table, if format_table is unspecified, otherwise vlines
%      is ignored.
%  mark_bold: String, one of {'none', 'min_of_row', 'max_of_row',
%      'min_of_col', 'max_of_col', '2ndmin_of_row', '2ndmax_of_row', ...}
%      Choose which table entries are highlighted using \textbf. Default:
%      'none'
%  mark_italic: String, one of {'none', 'min_of_row', 'max_of_row',
%      'min_of_col', 'max_of_col', '2ndmin_of_row', '2ndmax_of_row', ...}
%      Choose which table entries are highlighted using \textit. Default:
%      'none'
%  booktabs: Logical. If true, generate a table for use with package
%      booktabs.sty. Horizontal rules will be made with \toprule,
%      \midrule and \bottomrule instead of \hline. Using this option
%      overrides the values passed in options <vlines> (set to 'none')
%      and <hlines> (set to 'all'). Default value: 0
%   
% Description:
%   Include the table by '\input{<file>}' into your LaTeX document and
%   put the following macro definition in the preamble of your the LaTeX
%   file:
%      \newcommand{\columnTitle}[2]
%        {\multicolumn{#1}{@{\hspace*{0.5ex}}c@{\hspace{0.5ex}}}{#2}}
%
%   Known bugs: when you have a right vline and column titles.
%
% Examples:
%   a = [1.234 2.345;3.456 4.567];
%   latexTable(a, 1, 'mark_bold', 'min_of_row') will output on the terminal:
%      \begin{tabular}{@{}rr@{}}
%       \textbf{1.23} & 2.35\\
%       \textbf{3.46} & 4.57\\
%      \end{tabular}
%   latexTable(a, 'test, 'title', 'Test', 'col_title', {'col1', 'col2'})
%   will output to file test.tex:
%     \begin{tabular}{@{}l|rr@{}}
%      \textbf{Test}
%      & \columnTitle{1}{col1}
%      & \columnTitle{1}{col2}
%      \\ \hline
%       & 1.23 & 2.35\\
%       & 3.46 & 4.57\\
%     \end{tabular}
%   latexTable([1.2 1.3], 1, 'booktabs', 1, 'col_title', {'col1', 'col2'})
%   will output to terminal:
%     \begin{tabular}{@{}rr@{}}
%      \toprule 
%      \columnTitle{1}{col1}
%      & \columnTitle{1}{col2}
%      \midrule 
%      1.20 & 1.30\\
%      \bottomrule 
%     \end{tabular}
% 
% See also: fprintf,fopen,fclose
% 

% Author(s), Copyright: 
% Anton Schwaighofer, Jul 2005, based on latex_table from the BCI Toolbox
% by Benjamin Blankertz
% $Id: latexTable.m,v 1.4 2007/09/24 09:43:27 neuro_toolbox Exp $


error(nargchk(1, inf, nargin));
if nargin<2 | isempty(file),
  % Default: send to standard output
  file = 1;
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'title', [], ...
                  'col_title', [], ...
                  'table_type', 'tabular', ...
                  'mark_bold', 'none', ...
                  'mark_italic', 'none', ...
                  'format_entry', '%4.2f', ...
                  'col_summary', [], ...
                  'vlines', 'default', ...
                  'hlines', 'default', ...
                  'booktabs', 0, ...
                  'title_markup', '\textbf');
                  
if ~isfield(opt, 'row_title'),
  if isempty(opt.title),
    opt.row_title= [];
  else
    opt.row_title= repmat({''}, 1, size(data,1));
  end
end
if ischar(opt.format_entry),
  opt.format_entry= repmat({opt.format_entry}, 1, size(data,2));
end
if ~iscell(opt.format_entry),
  error('opt.format_entry must be a string or a cell array of strings');
end
if length(opt.format_entry)==size(data,2)+1,
  opt.format_entrytit= opt.format_entry{1};
  opt.format_entry= opt.format_entry(2:end);
elseif length(opt.format_entry)~=size(data,2),
  error('Option ''format_entry'' must match number of columns (or columns+1)');
else
  opt.format_entrytit= '%s';
end
if isempty(opt.row_title) & isempty(opt.title),
  %% in this case do not add a title column as first column
  rowtitsep= '';
else
  rowtitsep= ' &';
end
% For use with package booktabs: Override the vlines and hlines options
% brutally. Vertical lines will not work with booktabs because of the
% increased linespacing, and horizontal lines should be present above and
% below the table, as well as after the title
if opt.booktabs,
  opt.vlines = 'none';
  opt.hlines = 'all';
end
if ~isempty(opt.title_markup),
  % Strip trailing backslash
  while findstr(opt.title_markup, '\')==1,
    opt.title_markup = opt.title_markup(2:end);
  end
  while findstr(opt.title_markup, '{')==length(opt.title_markup),
    opt.title_markup = opt.title_markup(1:end-1);
  end
end
  
if ~isfield(opt, 'format_table'),
  opt.format_table= repmat('r',1,size(data,2));
  if ~isempty(rowtitsep),
    if strcmp(opt.vlines, 'none'),
      opt.format_table= ['l' opt.format_table];
    else
      opt.format_table= ['l|' opt.format_table];
    end
  end
  if ~isempty(strmatch(opt.vlines, {'all','left'})),
    opt.format_table= ['|' opt.format_table];
  else
    opt.format_table= ['@{}' opt.format_table];
  end
  if ~isempty(strmatch(opt.vlines, {'all','right'})),
    opt.format_table= [opt.format_table '|'];
  else
    opt.format_table= [opt.format_table '@{}'];
  end
end


mark_bold= make_marks(data, opt.mark_bold);
mark_italic= make_marks(data, opt.mark_italic);

if isnumeric(file),
  fid= file;
else
  [pathstr, filename, fileext] = fileparts(file);
  if isempty(fileext),
    fileext = '.tex';
  end
  file = fullfile(pathstr, [filename fileext]);
  [fid,message] = fopen(file, 'wt');
  if fid==-1,
    error(sprintf('Cannot open file %s for writing: %s', file, message));
  end
end
fprintf(fid, '\\begin{%s}{%s}\n', opt.table_type, opt.format_table);
if ~isempty(strmatch(opt.hlines, {'all','above'})),
  if opt.booktabs,
    fprintf(fid, ' \\toprule \n');
  else
    fprintf(fid, ' \\hline\n');
  end
end
if ~isempty(opt.title),
  if isempty(opt.title_markup),
    fprintf(fid, ' %s\n', opt.title);
  else
    fprintf(fid, ' \\%s{%s}\n', opt.title_markup, opt.title);
  end
end
if ~isempty(opt.col_title),
  nCols= size(data,2)/length(opt.col_title);
  if nCols~=round(nCols),
    error('length of opt.col_title must be a divisiors of number of columns');
  end
  fprintf(fid, rowtitsep);
  for cc= 1:length(opt.col_title),
    if cc>1,
      fprintf(fid, ' &');
    end
    fprintf(fid, ' \\columnTitle{%d}{%s}\n', nCols, opt.col_title{cc});
  end
end
if (~isempty(opt.title) | ~isempty(opt.col_title)) & ...
      ~strcmp(opt.hlines, 'none'),
  if opt.booktabs,
    fprintf(fid, ' \\\\ \n \\midrule \n');
  else
    fprintf(fid, ' \\\\ \n \\hline\n');
  end
end

for rr= 1:size(data,1),
  if ~isempty(opt.row_title),
    fprintf(fid, [' ' opt.format_entrytit rowtitsep], opt.row_title{rr});
  end
  latex_row(fid, data, rr, mark_bold, mark_italic, opt);
  fprintf(fid, '\\\\\n');
end

if ~isempty(opt.col_summary),
  if iscell(opt.col_summary),
    opt.col_summary_title= opt.col_summary{1};
    last_row=  opt.col_summary{2};
  else
    if ischar(opt.col_summary),
      if ~isfield(opt, 'col_summary_title'),
        opt.col_summary_title= opt.col_summary;
      end
      cleandata= data;  %% gehackt
      if strcmp(opt.col_summary, 'sum'),
        cleandata(find(isnan(data)))= 0;
      elseif strcmp(opt.col_summary, 'mean'),
        for cc= 1:size(data,2),
          nn= find(isnan(data(:,cc)));
          cleandata(nn,cc)= mean(data(find(~isnan(data(:,cc))),cc));
        end
      else
        error('hups?');
      end
      last_row= feval(opt.col_summary, cleandata, 1);
    else
      if ~isfield(opt, 'col_summary_title'),
        opt.col_summary_title= '';
      end
      last_row= opt.col_summary;
    end
  end
  if opt.booktabs,
    fprintf(fid, ' \\midrule \n');
  else
    fprintf(fid, ' \\hline\n');
  end
  if ~isempty(rowtitsep),
    fprintf(fid, [' \\textbf{%s}' rowtitsep], opt.col_summary_title);
  end
  if strpatterncmp('*row', opt.mark_bold),
    mark_bold= make_marks(last_row, opt.mark_bold);
  else
    mark_bold= zeros(size(last_row));
  end
  if strpatterncmp('*row', opt.mark_bold),
    mark_italic= make_marks(last_row, opt.mark_italic);
  else
    mark_italic= zeros(size(last_row));
  end
  latex_row(fid, last_row, 1, mark_bold, mark_italic, opt);
  fprintf(fid, '\\\\\n');
end

if ~isempty(strmatch(opt.hlines, {'all','below'})),
  if opt.booktabs,
    fprintf(fid, ' \\bottomrule \n');
  else
    fprintf(fid, ' \\hline\n');
  end
end
fprintf(fid, '\\end{%s}\n', opt.table_type);

if ischar(file),
  fclose(fid);
end

return




function latex_row(fid, data, rr, mark_bold, mark_italic, opt)

for cc= 1:size(data,2),
  if cc>1,
    fprintf(fid, ' &');
  end
  fprintf(fid, ' ');
  fmt= opt.format_entry{cc};
  if mark_bold(rr,cc),
    fmt= ['\\textbf{' fmt '}'];
  end
  if mark_italic(rr,cc),
    fmt= ['\\textit{' fmt '}'];
  end
  if iscell(data),
    fprintf(fid, fmt, data{rr,cc});
  else
    fprintf(fid, fmt, data(rr,cc));
  end
end

return




function marks= make_marks(data, policy)

switch(policy),
 case 'min_of_row',
  mm= min(data, [], 2);
  marks= (data==mm*ones(1,size(data,2)));
 case 'max_of_row',
  mm= max(data, [], 2);
  marks= (data==mm*ones(1,size(data,2)));
 case 'min_of_col',
  mm= min(data, [], 1);
  marks= (data==ones(size(data,1),1)*mm);
 case 'max_of_col',
  mm= max(data, [], 1);
  marks= (data==ones(size(data,1),1)*mm);
 case '2ndmin_of_row',
  mm= min(data, [], 2);
  ww= mm*ones(1,size(data,2));
  data(find(data==ww))= NaN;
  mm= min(data, [], 2);
  marks= (data==mm*ones(1,size(data,2)));
 case '2ndmax_of_row',
  mm= max(data, [], 2);
  ww= mm*ones(1,size(data,2));
  data(find(data==ww))= NaN;
  mm= max(data, [], 2);
  marks= (data==mm*ones(1,size(data,2)));
 case '2ndmin_of_col',
  mm= min(data, [], 1);
  ww= mm*ones(size(data,1),1);
  data(find(data==ww))= NaN;
  mm= min(data, [], 1);
  marks= (data==ones(size(data,1),1)*mm);
 case '2ndmax_of_col',
  mm= max(data, [], 1);
  ww= mm*ones(size(data,1),1);
  data(find(data==ww))= NaN;
  mm= max(data, [], 1);
  marks= (data==ones(size(data,1),1)*mm);
 case 'none',
  marks= zeros(size(data));
 otherwise,
  error(sprintf('<%s>: unknown mark policy', policy));
end
