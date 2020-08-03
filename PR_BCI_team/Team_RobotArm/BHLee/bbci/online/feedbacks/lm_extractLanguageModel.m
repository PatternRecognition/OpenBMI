function lm= lm_extractLanguageModel(file, varargin)
% LM_EXTRACTLANGUAGEMODEL - Build a language model from text files
%
%Synopsis:
% LM= lm_extractLanguageModel(FILE, <OPT>)
%
%Arguments:
% FILE: File name of input text file, or cell array of such.
%    When FILE is not an absolute path, [DATA_DIR 'language/source/']
%    is prepended.
% OPT: Struct or property/value list of opiotnal arguements
%  .saveas: Name to save the language model. This is always done in
%    [DATA_DIR 'language/']. If .saveas is empty the language model is
%    not saved.
%  .lm_headfactor: Vector of weights to weight word beginning ('head')
%     probabilities vs within word probabilities. The first value is
%     used as weight for the first letter of a word and so on. The last
%     value of the vector is used as weight for all subsequent letters.
%  .lm_npred: Number of letters for backward probability.
%  .lm_probdelete: Probability of delete symbol, default 0.1.
%
%Returns:
% LM: The extracted language model.
%
%See:
% lm_loadLanguageModel, hexlm_query, lm_getProbability

% Author(s): Benjamin Blankertz, Feb-2006

global DATA_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
		  'maxheadlen', 7, ...
		  'maxpredlen', 3, ...
		  'charset', [double('A'):double('Z'), '_.?'], ...
		  'saveas', '', ...
		  'language_model', []);

%% more internal stuff
opt= set_defaults(opt, ...
		  'table_growsize', 50);

if iscell(file),
  lm= [];
  for cc= 1:length(file),
    lm= lm_extractLanguageModel(file{cc}, varargin{:}, 'language_model',lm);
  end
  return;
end

nChars= length(opt.charset);
if isempty(opt.language_model),
  head_table= cell(1, opt.maxheadlen+1);
  head_prob= cell(1, opt.maxheadlen+1);
  for hl= 1:opt.maxheadlen+1,
    if hl>1,
      head_table{hl}= zeros(opt.table_growsize,hl-1);
      head_prob{hl}= zeros([nChars opt.table_growsize]);
    else
      head_prob{hl}= zeros([nChars 1]);
    end
  end
  pred_table= cell(1, opt.maxpredlen+1);
  pred_prob= cell(1, opt.maxpredlen+1);
  for hl= 1:opt.maxpredlen+1,
    if hl>1,
      pred_table{hl}= zeros(opt.table_growsize,hl-1);
      pred_prob{hl}= zeros([nChars opt.table_growsize]);
    else
      pred_prob{hl}= zeros([nChars 1]);
    end
  end
  [head_ptr{1:opt.maxheadlen+1}]= deal(0);
  [pred_ptr{1:opt.maxpredlen+1}]= deal(0);
else
  head_table= opt.language_model.head_table;
  head_prob= opt.language_model.head_prob;
  pred_table= opt.language_model.pred_table;
  pred_prob= opt.language_model.pred_prob;
  for hl= 1:opt.maxheadlen+1,
    head_ptr{hl}= size(head_table{hl},1);
  end
  for hl= 1:opt.maxpredlen+1,
    pred_ptr{hl}= size(pred_table{hl},1);
  end
end

%% Check for absolute paths:
%%  For Unix systems, absolute paths start with '\'.
%%  For Windoze, identify absolute paths by the ':' (e.g., H:\some\path).
if (isunix & (file(1)==filesep)) | (ispc & (file(2)==':')),
  fullname= file;
else
  fullname= [DATA_DIR 'language/source/' file];
end
fid= fopen(fullname);
if fid==-1,
  error('Cannot open text file.');
end

word_count= 0;
while ~feof(fid),
  tline= fgetl(fid);
  tline= replace_umlauts_etc(tline);
  tline= upper(tline);
  whitespace= find(~isletter(tline));
  tline(whitespace)= ' ';
  tline= [tline ' '];
  oldtline= '';
  while ~strcmp(tline,oldtline),
    oldtline= tline;
    tline= strrep(tline, '  ', ' ');
  end
  tline= strrep(tline, ' ', '_');
  wl= 0;
  for cc= 1:length(tline),
    cidx= find(tline(cc)==opt.charset);
    if isempty(cidx),
      fprintf('Unrecognized symbol ''%s'' in the following line:\n', ...
              tline(cc));
      fprintf('%s\n', tline);
      break;
    end
    if wl==0,
      head_prob{1}(cidx)= head_prob{1}(cidx) + 1;
    elseif wl<=opt.maxheadlen,
      head= tline(cc-wl:cc-1);
      idx= strmatch(head, head_table{wl+1}, 'exact');
      if isempty(idx),
        head_table{wl+1}= ...
            appendtotable(head, head_table{wl+1}, head_ptr{wl+1}, opt);
        [head_prob{wl+1}, head_ptr{wl+1}]= ...
            appendtoprob(zeros(nChars,1), head_prob{wl+1}, head_ptr{wl+1}, ...
                         opt);
        idx= head_ptr{wl+1};
      end
      head_prob{wl+1}(cidx,idx)= head_prob{wl+1}(cidx,idx) + 1;
    end
    pred_prob{1}(cidx)= pred_prob{1}(cidx) + 1;
    for bb= 1:min(wl, opt.maxpredlen),
      pred= tline(cc-bb:cc-1);
      idx= strmatch(pred, pred_table{bb+1}, 'exact');
      if isempty(idx),
        pred_table{bb+1}= ...
            appendtotable(pred, pred_table{bb+1}, pred_ptr{bb+1}, opt);
        [pred_prob{bb+1}, pred_ptr{bb+1}]= ...
            appendtoprob(zeros(nChars,1), pred_prob{bb+1}, pred_ptr{bb+1}, ...
                         opt);
        idx= pred_ptr{bb+1};
      end
      pred_prob{bb+1}(cidx,idx)= pred_prob{bb+1}(cidx,idx) + 1;
    end    
    wl= wl + 1;
    if tline(cc)=='_',
      wl= 0;
      word_count= word_count + 1;
      if not(mod(word_count,1000))
          fprintf('\r%7d words scanned.', word_count);
      end
    end
  end
end
fprintf('\n');

for hl= 1:opt.maxheadlen,
  head_table{hl+1}= head_table{hl+1}(1:head_ptr{hl+1}, :);
  head_prob{hl+1}= head_prob{hl+1}(:, 1:head_ptr{hl+1});
end
for hl= 1:opt.maxpredlen,
  pred_table{hl+1}= pred_table{hl+1}(1:pred_ptr{hl+1}, :);
  pred_prob{hl+1}= pred_prob{hl+1}(:, 1:pred_ptr{hl+1});
end


file_list= {file};
if ~isempty(opt.language_model),
  file_list= cat(2, opt.language_model.file_list, file_list);
end

if ~isempty(opt.saveas),
  global DATA_DIR
  opt= rmfield(opt, 'language_model');
  charset= opt.charset;
  opt= rmfields(opt, 'isPropertyStruct', 'table_growsize');
  save([DATA_DIR 'language/' opt.saveas], ...
       'head_table','head_prob','pred_table','pred_prob', 'charset', ...
       'file_list', 'opt');
  fprintf('language model save as <%s.mat>\n', ...
	  [DATA_DIR 'language/' opt.saveas]);
end

if nargout>=1,
  lm.head_table= head_table;
  lm.head_prob= head_prob;
  lm.pred_table= pred_table;
  lm.pred_prob= pred_prob;
  lm.charset= opt.charset;
  lm.file_list= file_list;
end



function [table, ptr]= appendtotable(entry, table, ptr, opt)

ptr= ptr + 1;
if ptr>size(table, 1),
  table= cat(1, table, zeros(opt.table_growsize, size(table,2)));
end
table(ptr,:)= entry;


function [table, ptr]= appendtoprob(entry, table, ptr, opt)

ptr= ptr + 1;
if ptr>size(table, 2),
  table= cat(2, table, zeros(size(table,1), opt.table_growsize));
end
table(:,ptr)= entry;
