function latex_table(file, table, fmt1, fmt2, col_title, row_title, ...
                     table_title);
%latex_table(file, table, fmt1, fmt2, <col_title, row_title, table_title>);
%
% instead of a file name, you can also use a file handle, e.g.,
% 1 for output on screen

if ~exist('col_title', 'var') | isempty(col_title),
  col_title= cell(1, size(table,2));
end
if ~exist('row_title', 'var') | isempty(row_title),
  col_title= cell(1, size(table,1));
end
if ~exist('table_title', 'var') | isempty(table_title),
  table_title= '';
end
nCols= size(table,2)/length(col_title);
if nCols~=round(nCols),
  error('length of col_title must be a divisiors of number of columns');
end


global TEX_DIR

if ischar(file) & file(1)~=filesep,
  file= [TEX_DIR file];
end

if isnumeric(file),
  fid= file;
else
  fid= fopen([file '.tex'], 'wt');
end
fprintf(fid, '\\begin{tabular}{%s}\n', fmt1);
if ~isempty(table_title),
  fprintf(fid, ' \\textbf{%s}\n', table_title);
end
for cc= 1:length(col_title),
  fprintf(fid, ' & \\columnTitle{%d}{%s}\n', nCols, col_title{cc});
end
fprintf(fid, ' \\\\ \\hline\n');

for rr= 1:size(table,1),
  fprintf(fid, [' ' fmt2 ' \\\\\n'], row_title{rr}, table(rr,:));
end
fprintf(fid, '\\end{tabular}\n');

if ischar(file),
  fclose(fid);
end
