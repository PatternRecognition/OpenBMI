function latex_bbci_results(paradigm, latex_file, varargin)
%latex_bbci_results(paradigm, latex_file, opt)
%
% IN  paradigm   - e.g. 'imag_motor'
%     latex_file - name of output file (TEX_DIR is preprended,
%                  if latex_file is not an absolute file name,
%                  i.e., beginning with '/'), default paradigm
%     opt - property/value list or struct
%      .methods  - 'all', or cell array of method names
%                  for each method name, a corresponding file
%                  [DATA_DIR 'results/' paradigm '/info_method_*'] must exist,
%                  default 'all'.
%      .sampling - cell array of sampling strategy names or
%                  vector of indices of sampling
%                  these refer to the variable 'sampling' in the file
%                  [DATA_DIR 'results/' paradigm '/info_paradigm'], or
%                  default 'all', which means all sampling methods for which
%                  a file
%                  [DATA_DIR 'results/' paradigm '/info_sampling_*'] exists
%      ...       - other fields are passed to latex_table

global TEX_DIR DATA_DIR

if ~exist('latex_file','var') | isempty(latex_file),
  latex_file= paradigm;
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'methods','all', ...
                  'sampling','all', ...
                  'datasets','all', ...
                  'header','portrait', ...
                  'rm_common_prefix', 1, ...
                  'row_summary', 'mean', ...
                  'mark_bold', 'min_of_row');

if latex_file(1)~='/',
  latex_file= strcat(TEX_DIR, latex_file);
end

sub_dir= [DATA_DIR 'results/' paradigm '/'];

if ischar(opt.methods) & strcmp(opt.methods, 'all'),
  dd= dir([sub_dir 'info_method_*.mat']);
  ll= length('info_method_');
  opt.methods= {dd.name};
  for ii= 1:length(opt.methods),
    opt.methods{ii}= opt.methods{ii}(ll+1:end-4);
  end
end
if opt.rm_common_prefix & length(opt.methods)>1,
  cc= char(opt.methods);
  ii= min(find(~all(cc==repmat(cc(1,:),[size(cc,1) 1]),1)));
  method_names= cell(1, length(opt.methods));
  for mm= 1:length(opt.methods),
    method_names{mm}= untex(opt.methods{mm}(ii:end));
  end
else
  method_names= apply_cellwise(opt.methods, 'untex');
end

if ischar(opt.sampling) & strcmp(opt.sampling, 'all'),
  dd= dir([sub_dir 'info_sampling_*.mat']);
  ll= length('info_sampling_');
  opt.sampling= {dd.name};
  for ii= 1:length(opt.sampling),
    opt.sampling{ii}= opt.sampling{ii}(ll+1:end-4);
  end
elseif isnumeric(opt.sampling),
  load([sub_dir 'info_paradigm'], 'sampling');
  opt.sampling= {sampling(opt.sampling).label};
end

load([sub_dir 'info_paradigm'], 'dataset');
if ischar(opt.datasets) & strcmp(opt.datasets, 'all'),
  opt.datasets= 1:length(dataset);
end

opt_tex= rmfield(opt, {'methods', 'sampling'});
opt_tex= set_defaults(opt_tex, ...
                      'fmt_row', '%.1f', ...
                      'table_type', 'longtable', ...
                      'col_title', method_names, ...
                      'row_title', apply_cellwise({dataset.label}, 'untex'));

fid= fopen([latex_file '.tex'], 'w');
if fid==-1,
  error(sprintf('cannot open <%s> for writing', latex_file));
end
header= textread([sub_dir paradigm '_header_' opt.header '.tex'],'%s', ...
                 'delimiter','\n', 'whitespace','');
fprintf(fid, '%s\n', header{:});
fprintf(fid, '\\title{Classification Results of `%s'' Experiments}\n', ...
        untex(paradigm));
[dmy,mylogin]= system('whoami');
fprintf(fid, '\\author{%s}\n', mylogin);
fprintf(fid, '\\maketitle\n\n');

fprintf(fid, 'The following methods are compared:\n');
fprintf(fid, '\\begin{description}\n');
for mm= 1:length(opt.methods),
  load([sub_dir 'info_method_' opt.methods{mm}], 'desc');
  fprintf(fid, ' \\item[%s] %s\n', method_names{mm}, untex(desc));
end
fprintf(fid, '\\end{description}\n\n');

Loss= NaN*zeros(length(opt.datasets), length(opt.methods));
Loss_std= NaN*zeros(length(opt.datasets), length(opt.methods));
for si= 1:length(opt.sampling),
  sampling_label= opt.sampling{si};
  load([sub_dir 'info_sampling_' sampling_label], 'sampling');
  if length(opt.sampling)>1,
    if length(opt.datasets)>20,
      fprintf(fid, '\n\\pagebreak\n\n');
    end
    fprintf(fid, '\\section{Validation by %s}\n\n', untex(sampling.label));
  end
  fprintf(fid, 'The following strategy was used to validate the methods: ');
  fprintf(fid, '%s\n\\bigskip\n\n', untex(sampling.desc));
  for mm= 1:length(opt.methods),
    for di= 1:length(opt.datasets),
      dd= opt.datasets(di);
      results_file= [sub_dir dataset(dd).label '/result_' opt.methods{mm} ...
                     '_sampling_' sampling_label];
      if exist([results_file '.mat'], 'file'),
        R= load(results_file);
        Loss(di,mm)= R.loss;
        Loss_std(di,mm)= R.loss_std;
      else
        Loss(di,mm) = NaN;
        Loss_std(di,mm) = NaN;
      end
    end
  end
  iNoResult= find(all(isnan(Loss),2));
  Loss(iNoResult,:)= [];
  opt_tex.row_title= apply_cellwise({dataset.label}, 'untex');
  opt_tex.row_title(iNoResult)= [];
  opt_tex.title= untex(sampling_label);
  latex_table(fid, 100*Loss, opt_tex);
  fprintf(fid, '\n\n');
end

fprintf(fid, '\\end{document}\n');
fclose(fid);
