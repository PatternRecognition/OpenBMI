function s= construct(entrylist, varargin)
%s= construct(entrylist, field1, field2, ...)

nFields= nargin-1;
constructor= {};
for fi= 1:nFields,
  constructor= {constructor{:}, varargin{fi}, {entrylist{fi:nFields:end}}};
end

s= struct(constructor{:}); 
