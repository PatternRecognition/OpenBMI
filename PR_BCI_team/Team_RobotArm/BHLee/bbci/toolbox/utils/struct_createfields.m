function s= struct_createfields(s, flds, varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...                 
    set_defaults(opt, ...
                 'matchsize', [], ...
                 'value', NaN);

if ischar(flds),
  flds= {flds};
end

for ii= 1:length(flds),
  if isempty(opt.matchsize),
    val= opt.value;
  else
    val= repmat(opt.value, size(opt.matchsize.(flds{ii})));
  end
  [s.(flds{ii})]= deal(val);
end
