function params = gaussProc_removeClamped(params, clamped)
% gaussProc_removeClamped - GP helper function: Remove clamped parameters from a list of allowed parameters
%
% Synopsis:
%   params = gaussProc_removeClamped(params,clamped)
%   

for i = 1:length(clamped),
  c = clamped{i};
  if (iscell(c) & length(c)==1 & ischar(c{1})),
    % Entry is of the form {'fieldname'}
    params = removeField(params, c{1});
  elseif ischar(c),
    % Entry is of the form 'fieldname'
    params = removeField(params, c);
  elseif (iscell(c) & length(c)==2 & ischar(c{1}) & isnumeric(c{2})),
    % Entry is of the form {'fieldname', index}
    remove = logical(zeros(size(params)));
    for j = 1:length(params),
      p = params{j};
      % Look whether we find a similar entry {'fieldname', index}
      if iscell(p) & length(p)==2 & ischar(p{1}) & isnumeric(p{2}),
        remove(i) = strcmp(c{1}, p{1}) & (c{2}==p{2});
      end
    end
    params = params(~remove);
  else
    error(sprintf('Invalid entry in clamped{%i}', i));
  end
end


function params = removeField(params, f)

remove = logical(zeros(size(params)));
for i = 1:length(params),
  p = params{i};
  if iscell(p),
    if (length(p)==1 & ischar(p{1})) | ...
          (length(p)==2 & ischar(p{1}) & isnumeric(p{2})),
      % Entry is of the form {'fieldname'} or {'fieldname', index}
      remove(i) = strcmp(f, p{1});
    else
      error(sprintf('Invalid entry in params{%i}', i));
    end
  elseif ischar(p),
    % Entry is of the form 'fieldname'
    remove(i) = strcmp(f, p);
  else
    error(sprintf('Invalid entry in fields{%i}', i));
  end
end
params = params(~remove);
