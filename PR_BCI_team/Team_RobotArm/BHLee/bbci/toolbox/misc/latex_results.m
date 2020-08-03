function latex_results(paradigm, latex_file, varargin)

global TEX_DIR DATA_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'methods','*', ...
                  'header','portrait', ...
                  'rm_common_prefix', 1, ...
                  'rm_common_postfix', 1, ...
                  'row_summary', 'mean', ...
                  'skip_rows', [], ...
                  'mark_bold', 'min_of_row', ...
                  'mark_italic', '2ndmin_of_row', ...
                  'performance_variable', 'perf', ...
                  'performance_factor', 1, ...
                  'itemize_methods', 1, ...
                  'latex', 'pdf');

section_title= latex_file;
if latex_file(1)~='/',
  latex_file= strcat(TEX_DIR, latex_file);
end

sub_dir= [DATA_DIR 'results/' paradigm '/'];

if ~iscell(opt.methods),
  opt.methods= {opt.methods};
end
dd= dir([sub_dir '*.mat']);
all_files= {dd.name};
for ii= 1:length(all_files),
  all_files{ii}= all_files{ii}(1:end-4);
end
meth_idx= strpatternmatch(opt.methods, all_files);
if ~isequal(unique(opt.methods),unique(all_files(meth_idx))),
  opt.methods= all_files(meth_idx);
end
method_names= opt.methods;
if opt.rm_common_prefix & length(method_names)>1,
  cc= char(method_names);
  ii= min(find(~all(cc==repmat(cc(1,:),[size(cc,1) 1]),1)));
  for mm= 1:length(method_names),
    method_names{mm}= method_names{mm}(ii:end);
  end
end
if opt.rm_common_postfix & length(method_names)>1,
  cc= char(apply_cellwise(method_names,'fliplr'));
  ii= min(find(~all(cc==repmat(cc(1,:),[size(cc,1) 1]),1)));
  for mm= 1:length(method_names),
    method_names{mm}= method_names{mm}(1:end-ii+1);
  end
end
method_names= apply_cellwise(method_names, 'untex');
if opt.itemize_methods,
  numbers= unblank(cellstr(int2str([1:length(method_names)]')));
  method_tags= strcat('(', numbers, ')');
else
  method_tags= method_names;
end

R= load([sub_dir opt.methods{1}]);
perf= getfield(R, opt.performance_variable);
nDatasets= length(perf(:));

opt_tex= rmfield(opt, {'methods'});
opt_tex= set_defaults(opt_tex, ...
                      'fmt_row', '%.1f', ...
                      'table_type', 'longtable', ...
                      'col_title', method_tags);
if isempty(opt.dataset_names),
  opt.dataset_names= cprintf('%d', 1:nDatasets);
end
opt_tex.row_title= apply_cellwise(opt.dataset_names, 'untex');

fid= fopen([latex_file '.tex'], 'w');
if fid==-1,
  error(sprintf('cannot open <%s.tex> for writing', latex_file));
end
header_file= [sub_dir paradigm '_header_' opt.header '.tex'];
if ~exist(header_file, 'file'),
  header_file= [DATA_DIR 'results/header_' opt.header '.tex'];
end
header= textread(header_file,'%s', ...
                 'delimiter','\n', 'whitespace','');
fprintf(fid, '%s\n', header{:});
fprintf(fid, '\\title{Classification Results of `%s'' Experiments}\n', ...
        untex(paradigm));
[dmy,mylogin]= system('whoami');
fprintf(fid, '\\author{%s}\n', mylogin);
fprintf(fid, '\\maketitle\n\n');

if opt.itemize_methods,
  fprintf(fid, 'The following methods are compared:\n');
  fprintf(fid, '\\begin{enumerate}\n');
  for mm= 1:length(method_names),
    fprintf(fid, ' \\item %s\n', method_names{mm});
  end
  fprintf(fid, '\\end{enumerate}\n\n');
end
fprintf(fid, '\\section{%s}\n', untex(section_title));
if opt.itemize_methods,
  if strpatterncmp('*landscape*', opt.header),
    fprintf(fid, '\n\\pagebreak\n\n');
  end
else
  %% a hack to get space for rotated col titles
  fprintf(fid, '\\vspace*{4cm}\n\n');
end

Loss= NaN*zeros(nDatasets, length(opt.methods));
for mm= 1:length(opt.methods),
  results_file= [sub_dir opt.methods{mm}];
  if exist([results_file '.mat'], 'file'),
    R= load(results_file);
    perf= getfield(R, opt.performance_variable);
    Loss(:,mm)= perf(:);
  end
end
iNoResult= find(all(isnan(Loss),2));
iSkip= union(iNoResult, opt.skip_rows);
Loss(iSkip,:)= [];
opt_tex.row_title(iSkip)= [];
latex_table(fid, opt.performance_factor*Loss, opt_tex);
fprintf(fid, '\n\n');

fprintf(fid, '\\end{document}\n');
fclose(fid);

[filepath,filename]= fileparts(latex_file);
if strcmpi(opt.latex,'pdf'),
  cmd= sprintf('cd %s; LD_LIBRARY_PATH="" pdflatex %s; chmod g=u %s.*', ...
               filepath, filename, filename);
  unix(cmd);
elseif strcmpi(opt.latex, 'ps'),
  if strpatterncmp('*landscape*', opt.header),
    dviopt= '-t landscape';
  else
    dviopt= '';
  end
  cmd= sprintf('cd %s; latex %s; dvips %s %s; chmod g=u %s.*', ...
               filepath, filename, dviopt, filename, filename);
  unix(cmd);
end
