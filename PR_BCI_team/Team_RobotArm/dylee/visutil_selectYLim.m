function YLim= visutil_selectYLim(h, varargin)

props = {'Policy',          'auto'          '!CHAR(auto tightest tight)';
         'TightenBorder'     0.03        	'DOUBLE';
         'Symmetrize'       0               '!BOOL';
         'SetLim'           1               '!BOOL';
         };

if nargin==0,
  YLim= props; return
end

opt= opt_proplistToStruct(varargin{:});
[opt, isdefault]= opt_setDefaults(opt, props);
opt_checkProplist(opt, props);
misc_checkType(h,'!GRAPHICS');

switch(opt.Policy),
 case 'auto',
  YLim= get(h, 'YLim');
 case 'tightest',
  visutil_backaxes(h);
  axis('tight');
  YLim= get(h, 'YLim');
 case 'tight',
  visutil_backaxes(h);
  axis('tight');
  yl= get(h, 'YLim');
  % add border not to make it too tight:
  yl= yl + [-1 1]*opt.TightenBorder*diff(yl);
  % determine nicer limits
  dig= floor(log10(diff(yl)));
  if diff(yl)>1,
    dig= max(1, dig);
  end
  YLim= [util_trunc(yl(1),-dig+1,'floor') util_trunc(yl(2),-dig+1,'ceil')];
end

if opt.Symmetrize,
  ma= max(abs(YLim));
  YLim= [-ma ma];
end

if opt.SetLim,
  set(h, 'YLim',YLim);
end

if nargout==0,
  clear YLim;
end