function fig_setDimensions(varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'ratio', '', ...
                 'absolute', '', ...
                 'units', 'normalized', ...
                 'policy', 'change_both');

if ~isdefault.ratio & ~isdefault.absolute,
  error('you may only specify one of the properties RATIO or ABSOLUTE');
end

oldUnits= get(gcf, 'Units');

if ~isdefault.absolute,
  set(gcf, 'Units',opt.units);
  pos= get(gcf, 'Position');
  pos(3:4)= opt.absolute;
elseif ~isdefault.ratio,
  set(gcf, 'Units','Points');
  pos= get(gcf, 'Position');
  if ischar(opt.ratio),
    is= find(opt.ratio==':');
    if length(is)~=1,
      error('if value of RATIO is a char, it must be of format <xx:yy>');
    end
    xr= str2double(opt.ratio(1:is-1));
    yr= str2double(opt.ratio(is+1:end));
    opt.ratio= xr/yr;
  end
  switch(lower(opt.policy)),
   case 'change_x',
    pos(3)= pos(4)*opt.ratio;
   case 'change_y',
    pos(4)= pos(3)/opt.ratio;
   case 'change_both',
    error('not implemented');
   otherwise,
    error('policy not known');
  end
end

set(gcf, 'Position',pos);
set(gcf, 'Units',oldUnits);
