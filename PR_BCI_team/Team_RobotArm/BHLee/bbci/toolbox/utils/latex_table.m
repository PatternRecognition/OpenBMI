function latex_table(file, table, varargin);
%latex_table(file, table, <property_list>)
%latex_table(file, table, <opt>)
%
% output a matlab matrix as LaTeX tabular code.
%
% IN  file  - name of output latex file (without extension '.tex')
%             when no absolute name is given, global EEG_TEX_DIR is prepended
%     table - matrix that should be converted
%     opt
%        .fmt_row     - cell array, defining the print format for one row.
%                       the length must be equal to the number of columns.
%                       or it can be a string, when all entries should be
%                       formatted in the same way. default '%4.2f'.
%        .fmt_table   - default '@{}l|r...r@{}' with #cols many r's.
%        .col_title   - cell array of strings, is put in the first row. 
%                       the length must be a divisor of #cols. when < #cols,
%                       each title is heading several columns, namely
%                       length(col_title)/nCols many.
%        .row_title   - cell array of strings, is put on in the first column.
%                       then length must equal #rows.
%        .title       - string, is put in the upper left corner.
%        .table_type  - e.g. 'tabular' (default), 'longtable'
%        .row_summary - {'mean', 'sum'}
%        .hlines      - {'all', 'above', 'below', 'none', 'default'},
%                       'default' is one line below the row of col titles.
%                       remark: vertical lines can be controlled with
%                       field 'fmt_table'.
%        .mark_bold   - {'none', 'min_of_row', 'max_of_row',
%                        'min_of_col', 'max_of_col', '2ndmin_of_row', ...}
%        .mark_italic - as mark_bold
%
% you can give the options as struct or as paired property list, or
% as a mixture of both, e.g.
%  opt= struct('mark_bold', 'min_of_row');
%  latex_table(1, randn(5,3), opt, 'title','zufall');
%
% instead of a file name, you can also use a file handle, e.g.,
% 1 for output on screen
%
% include the table by '\input{<file>}' into your latex document and
% put the following macro definition in the preamble of your latex file:
%  \newcommand{\columnTitle}[2]
%   {\multicolumn{#1}{@{\hspace*{0.5ex}}c@{\hspace{0.5ex}}}{#2}}
%
% known bugs: when you have a right vline and column titles.

% bb, ida.first.fhg.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'title', [], ...
                  'col_title', [], ...
                  'col_title_rot', 0, ...
                  'table_type', 'tabular', ...
                  'mark_bold', 'none', ...
                  'mark_italic', 'none', ...
                  'fmt_row', '%4.2f', ...
                  'hlines', 'default', ...
                  'vlines', 'default');
                  
if ~isfield(opt, 'row_title'),
  if isempty(opt.title),
    opt.row_title= [];
  else
    opt.row_title= repmat({''}, 1, size(table,1));
  end
end
if ischar(opt.fmt_row),
  opt.fmt_row= repmat({opt.fmt_row}, 1, size(table,2));
end
if ~iscell(opt.fmt_row),
  error('opt.fmt_row must be a string or a cell array of strings');
end
if length(opt.fmt_row)==size(table,2)+1,
  opt.fmt_rowtit= opt.fmt_row{1};
  opt.fmt_row= opt.fmt_row(2:end);
else
  opt.fmt_rowtit= '%s';
end
if isempty(opt.row_title) & isempty(opt.title),
  %% in this case do not add a title column as first column
  rowtitsep= '';
else
  rowtitsep= ' &';
end
if ~isfield(opt, 'fmt_table'),
  opt.fmt_table= repmat('r',1,size(table,2));
  if ~isempty(rowtitsep),
    if strcmp(opt.vlines, 'none'),
      opt.fmt_table= ['l' opt.fmt_table];
    else
      opt.fmt_table= ['l|' opt.fmt_table];
    end
  end
  if ~isempty(strmatch(opt.vlines, {'all','left'})),
    opt.fmt_table= ['|' opt.fmt_table];
  else
    opt.fmt_table= ['@{}' opt.fmt_table];
  end
  if ~isempty(strmatch(opt.vlines, {'all','right'})),
    opt.fmt_table= [opt.fmt_table '|'];
  else
    opt.fmt_table= [opt.fmt_table '@{}'];
  end
end


global TEX_DIR

mark_bold= make_marks(table, opt.mark_bold);
mark_italic= make_marks(table, opt.mark_italic);

if ischar(file) & file(1)~=filesep,
  file= [TEX_DIR file];
end

if isnumeric(file),
  fid= file;
else
  fid= fopen([file '.tex'], 'wt');
  if fid==-1,
    error(sprintf('cannot open %s for writing', [file '.tex']));
  end
end
fprintf(fid, '\\begin{%s}{%s}\n', opt.table_type, opt.fmt_table);
if ~isempty(strmatch(opt.hlines, {'all','above'})),
  fprintf(fid, ' \\hline\n');
end
if ~isempty(opt.title),
  fprintf(fid, ' \\textbf{%s}\n', opt.title);
end
if ~isempty(opt.col_title),
  nCols= size(table,2)/length(opt.col_title);
  if nCols~=round(nCols),
    error('length of opt.col_title must be a divisiors of number of columns');
  end
  fprintf(fid, rowtitsep);
  for cc= 1:length(opt.col_title),
    if cc>1,
      fprintf(fid, ' &');
    end
    if opt.col_title_rot,
      fprintf(fid, ' \\columnTitleRot{%d}{%s}\n', nCols, opt.col_title{cc});
    else
      fprintf(fid, ' \\columnTitle{%d}{%s}\n', nCols, opt.col_title{cc});
    end
  end
end
if (~isempty(opt.title) | ~isempty(opt.col_title)) & ...
      ~strcmp(opt.hlines, 'none'),
  fprintf(fid, ' \\\\ \\hline\n');
end

for rr= 1:size(table,1),
  if ~isempty(opt.row_title),
    fprintf(fid, [' ' opt.fmt_rowtit rowtitsep], opt.row_title{rr});
  end
  latex_row(fid, table, rr, mark_bold, mark_italic, opt);
  fprintf(fid, '\\\\\n');
end

if isfield(opt, 'row_summary'),
  if iscell(opt.row_summary),
    opt.row_summary_title= opt.row_summary{1};
    last_row=  opt.row_summary{2};
  else
    if ischar(opt.row_summary),
      if ~isfield(opt, 'row_summary_title'),
        opt.row_summary_title= opt.row_summary;
      end
%      cleantable= table;  %% gehackt [? -> use nansum, nanmean?]
%      if strcmp(opt.row_summary, 'sum'),
%        cleantable(find(isnan(table)))= 0
%      elseif strcmp(opt.row_summary, 'mean'),
%        for cc= 1:size(table,2),
%          nn= find(isnan(table(:,cc)));
%          cleantable(nn,cc)= mean(table(find(~isnan(table(:,cc))),cc));
%        end
%      else
%        error('hups?');
%      end
      if any(isnan(table(:))),
        rowsummary= ['nan' opt.row_summary];
      else
        rowsummary= opt.row_summary;
      end
      last_row= feval(rowsummary, table, 1);
    else
      if ~isfield(opt, 'row_summary_title'),
        opt.row_summary_title= '';
      end
      last_row= opt.row_summary;
    end
  end
  fprintf(fid, ' \\hline\n');
  if ~isempty(rowtitsep),
    fprintf(fid, [' \\textbf{%s}' rowtitsep], opt.row_summary_title);
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
  fprintf(fid, ' \\hline\n');
end
fprintf(fid, '\\end{%s}\n', opt.table_type);

if ischar(file),
  fclose(fid);
end

return




function latex_row(fid, table, rr, mark_bold, mark_italic, opt)

for cc= 1:size(table,2),
  if cc>1,
    fprintf(fid, ' &');
  end
  fprintf(fid, ' ');
  fmt= opt.fmt_row{cc};
  if mark_bold(rr,cc),
    fmt= ['\\textbf{' fmt '}'];
  end
  if mark_italic(rr,cc),
    fmt= ['\\textit{' fmt '}'];
  end
  fprintf(fid, fmt, table(rr,cc));
end

return




function marks= make_marks(table, policy)

switch(policy),
 case 'min_of_row',
  mm= min(table, [], 2);
  marks= (table==mm*ones(1,size(table,2)));
 case 'max_of_row',
  mm= max(table, [], 2);
  marks= (table==mm*ones(1,size(table,2)));
 case 'min_of_col',
  mm= min(table, [], 1);
  marks= (table==ones(size(table,1),1)*mm);
 case 'max_of_col',
  mm= max(table, [], 1);
  marks= (table==ones(size(table,1),1)*mm);
 case '2ndmin_of_row',
  mm= min(table, [], 2);
  ww= mm*ones(1,size(table,2));
  table(find(table==ww))= NaN;
  mm= min(table, [], 2);
  marks= (table==mm*ones(1,size(table,2)));
 case '2ndmax_of_row',
  mm= max(table, [], 2);
  ww= mm*ones(1,size(table,2));
  table(find(table==ww))= NaN;
  mm= max(table, [], 2);
  marks= (table==mm*ones(1,size(table,2)));
 case '2ndmin_of_col',
  mm= min(table, [], 1);
  ww= mm*ones(size(table,1),1);
  table(find(table==ww))= NaN;
  mm= min(table, [], 1);
  marks= (table==ones(size(table,1),1)*mm);
 case '2ndmax_of_col',
  mm= max(table, [], 1);
  ww= mm*ones(size(table,1),1);
  table(find(table==ww))= NaN;
  mm= max(table, [], 1);
  marks= (table==ones(size(table,1),1)*mm);
 case 'none',
  marks= zeros(size(table));
 otherwise,
  error(sprintf('<%s>: unknown mark policy', policy));
end
