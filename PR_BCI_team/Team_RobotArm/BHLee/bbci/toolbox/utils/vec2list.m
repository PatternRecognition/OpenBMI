function varargout= vec2list(ve)

for ii= 1:min(length(ve),nargout),
  if iscell(ve),
    varargout{ii}= ve{ii};
  else
    varargout{ii}= ve(ii);
  end
end
