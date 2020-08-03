function [subject, date_str, paradigm, appendix]= ...
    expbase_decomposeFilename(file, sub_dir)

if iscell(file),
  nFiles= length(file);
  subject= cell(1, nFiles);
  date_str= cell(1, nFiles);
  paradigm= cell(1, nFiles);
  appendix= cell(1, nFiles);
  for fn= 1:nFiles,
    [subject{fn}, date_str{fn}, paradigm{fn}, appendix{fn}]= ...
        expbase_decomposeFilename(file{fn});
  end
  if length(unique(subject))==1, subject= subject{1}; end
  if length(unique(date_str))==1, date_str= date_str{1}; end
  if length(unique(paradigm))==1, paradigm= paradigm{1}; end
  if length(unique(appendix))==1, appendix= appendix{1}; end
  return;
end

is= find(file=='/');
if length(is)>1,
  file(1:is(end-1))= [];
  is= is(end);
end
if isempty(is),
  if nargin<2,
    error('file must contain subdirectory, or subdir must be given');
  end
  file_name= file;
else
  sub_dir= file(1:is-1);
  file_name= file(is+1:end);
end
iu= min(find(sub_dir=='_'));
subject= sub_dir(1:iu-1);
date_str= sub_dir(iu+1:end);
is= min([findstr(file_name, subject), length(file_name)-3]);
exp_name= file_name(1:is-1);
nn= length(exp_name);
lastnondigit= length(exp_name);
while ismember(exp_name(lastnondigit), ['0':'9']),
  lastnondigit= lastnondigit-1;
end
idigit= find(ismember(exp_name, ['0':'9']));
couldbesec= find(exp_name=='s');
issec= intersect(couldbesec, idigit+1);
if ~isempty(issec),
  nn= issec-1;
  while ismember(exp_name(nn), ['0':'9', '_']),
    nn= nn-1;
  end
elseif strcmp('fb',exp_name(lastnondigit-1:lastnondigit)) ...
      | strcmp('ft',exp_name(lastnondigit-1:lastnondigit)),
  nn= lastnondigit-2;
  while exp_name(nn)~='_',
    nn= nn-1;
  end
  nn= nn-1;
elseif ~isempty(findstr('_var', exp_name)),
  nn= max(findstr('_var', exp_name))-1;
end
paradigm= exp_name(1:nn);
if nn<length(exp_name)
  appendix= exp_name(nn+1:end);
else
  appendix= '';  %% otherwise it would be [1x0 char]
end
