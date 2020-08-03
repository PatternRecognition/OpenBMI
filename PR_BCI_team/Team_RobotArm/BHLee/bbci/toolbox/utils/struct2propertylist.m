function pl= struct2propertylist(opt)

if isempty(opt),
  pl= {};
else
  C= cat(2, fieldnames(opt), struct2cell(opt))';
  pl= C(:)';
end

%if nargout==1,
%  varargout= {pl};
%else
%  varargout= cell(1, length(pl));
%  [varargout{:}]= deal(pl{:});
%end
