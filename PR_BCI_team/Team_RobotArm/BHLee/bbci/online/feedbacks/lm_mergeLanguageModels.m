function lm= lm_mergeLanguageModels(lms, varargin)
% LM_MERGELANGUAGEMODEL - Merge several language models
%
%Synopsis:
% LM= lm_mergeLanguageModels(LMS, <OPT>);
%
%Arguments:
% LMS: Cell array of language models (either file names of language models
%    = cell array of strings, or language models itself = cell array of
%    structs). In the latter case, be sure to use the 'raw' language
%    models, not the ones with probabilities which you get from
%    lm_loadLanguageModel.
% OPT: Struct or property/value list of opiotnal arguements
%  'saveas': Name to save the language model. This is always done in
%    [DATA_DIR 'language/']. If .saveas is empty the language model is
%    not saved.
%  'factor': Vector of factors (length matching the length of LMS)
%    with which the occurence counters of the respective language model 
%    are multiplied.
%
%Returns:
% LM: The merged language model.
%
%See:
% lm_extractLanguageModel, lm_loadLanguageModel, hexlm_query

% Author(s): Benjamin Blankertz, Mar-2006

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'factor',ones(1, length(lms)), ...
                  'table_growsize', 50, ...
                  'saveas', []);

global DATA_DIR

%% language models can be specified as file names
if ischar(lms{1}),
  for kk= 1:length(lms),
    lms{kk}= load([DATA_DIR 'language/' lms{kk}]);
  end
end

lm= lms{1};
lm= lm_multiply(lm, opt.factor(1));
for kk= 2:length(lms),
  lm2= lm_multiply(lms{kk}, opt.factor(kk));
  lm= lm_mergeTwo(lm, lm2, opt);
end

if ~isempty(opt.saveas),
  global DATA_DIR
  head_table= lm.head_table;
  head_prob= lm.head_prob;
  pred_table= lm.pred_table;
  pred_prob= lm.pred_prob;
  charset= lm.charset;
  file_list= lm.file_list;
  opt= setfield(lm.opt, 'saveas',opt.saveas);
  save([DATA_DIR 'language/' opt.saveas], ...
       'head_table','head_prob','pred_table','pred_prob', 'charset', ...
       'file_list', 'opt');
  fprintf('language model save as <%s.mat>\n', ...
	  [DATA_DIR 'language/' opt.saveas]);
end

return;


%% ---------------


function lm= lm_multiply(lm, factor)

for hl= 1:length(lm.head_prob),
  lm.head_prob{hl}= factor * lm.head_prob{hl};
end
for hl= 1:length(lm.pred_prob),
  lm.pred_prob{hl}= factor * lm.pred_prob{hl};
end
if ~isfield(lm, 'opt') | ~isfield(lm.opt, 'factor'),
  lm.opt.factor= ones(1, length(lm.file_list));
end
lm.opt.factor= factor * lm.opt.factor;

return;


%% ---------------


function lm= lm_mergeTwo(lm, lm2, opt)

if ~isequal(lm.charset, lm2.charset),
  %% One could try to match them ...
  error('charsets do not match');
end
nChars= length(lm.charset);

lm.file_list= cat(2, lm.file_list, lm2.file_list);
lm.opt.factor= cat(2, lm.opt.factor, lm2.opt.factor);

if lm.opt.maxheadlen ~= lm2.opt.maxheadlen,
  warning('maxheadlen does not match: truncating to min');
  lm.opt.maxheadlen= min(lm.opt.maxheadlen, lm2.opt.maxheadlen);
  lm.pred_prob= lm.pred_prob(1:lm.opt.maxheadlen+1);
  lm.pred_table= lm.pred_table(1:lm.opt.maxheadlen+1);
  lm2.pred_prob= lm2.pred_prob(1:lm.opt.maxheadlen+1);
  lm2.pred_table= lm2.pred_table(1:lm.opt.maxheadlen+1);
end
if lm.opt.maxpredlen ~= lm2.opt.maxpredlen,
  warning('maxpredlen does not match: truncating to min');
  lm.opt.maxpredlen= min(lm.opt.maxpredlen, lm2.opt.maxpredlen);
  lm.pred_prob= lm.pred_prob(1:lm.opt.maxpredlen+1);
  lm.pred_table= lm.pred_table(1:lm.opt.maxpredlen+1);
  lm2.pred_prob= lm2.pred_prob(1:lm.opt.maxpredlen+1);
  lm2.pred_table= lm2.pred_table(1:lm.opt.maxpredlen+1);
end
for hl= 1:lm.opt.maxheadlen+1,
  head_ptr{hl}= size(lm.head_table{hl},1);
end
for hl= 1:lm.opt.maxpredlen+1,
  pred_ptr{hl}= size(lm.pred_table{hl},1);
end

lm.head_prob{1}= lm.head_prob{1} + lm2.head_prob{1};
for hl= 2:length(lm2.head_prob),
  for ii= 1:size(lm2.head_prob{hl},2),
    head= lm2.head_table{hl}(ii,:);
    idx= strmatch(head, lm.head_table{hl}, 'exact');
    if isempty(idx),
      lm.head_table{hl}= ...
          appendtotable(head, lm.head_table{hl}, head_ptr{hl}, opt);
      [lm.head_prob{hl}, head_ptr{hl}]= ...
          appendtoprob(zeros(nChars,1), lm.head_prob{hl}, ...
                       head_ptr{hl}, opt);
      idx= head_ptr{hl};
    end
    lm.head_prob{hl}(:,idx)= lm.head_prob{hl}(:,idx) + ...
        lm2.head_prob{hl}(:,ii);
  end
end

lm.pred_prob{1}= lm.pred_prob{1} + lm2.pred_prob{1};
for hl= 2:length(lm2.pred_prob),
  for ii= 1:size(lm2.pred_prob{hl},2),
    pred= lm2.pred_table{hl}(ii,:);
    idx= strmatch(pred, lm.pred_table{hl}, 'exact');
    if isempty(idx),
      lm.pred_table{hl}= ...
          appendtotable(pred, lm.pred_table{hl}, pred_ptr{hl}, opt);
      [lm.pred_prob{hl}, pred_ptr{hl}]= ...
          appendtoprob(zeros(nChars,1), lm.pred_prob{hl}, ...
                       pred_ptr{hl}, opt);
      idx= pred_ptr{hl};
    end
    lm.pred_prob{hl}(:,idx)= lm.pred_prob{hl}(:,idx) + ...
        lm2.pred_prob{hl}(:,ii);
  end
end

for hl= 2:length(lm.head_prob),
  lm.head_table{hl}= lm.head_table{hl}(1:head_ptr{hl}, :);
  lm.head_prob{hl}= lm.head_prob{hl}(:, 1:head_ptr{hl});
end
for hl= 2:length(lm.pred_prob),
  lm.pred_table{hl}= lm.pred_table{hl}(1:pred_ptr{hl}, :);
  lm.pred_prob{hl}= lm.pred_prob{hl}(:, 1:pred_ptr{hl});
end

return;


%% ---------------

function [table, ptr]= appendtotable(entry, table, ptr, opt)

ptr= ptr + 1;
if ptr>size(table, 1),
  table= cat(1, table, zeros(opt.table_growsize, size(table,2)));
end
table(ptr,:)= entry;

return;


%% ---------------

function [table, ptr]= appendtoprob(entry, table, ptr, opt)

ptr= ptr + 1;
if ptr>size(table, 2),
  table= cat(2, table, zeros(size(table,1), opt.table_growsize));
end
table(:,ptr)= entry;
