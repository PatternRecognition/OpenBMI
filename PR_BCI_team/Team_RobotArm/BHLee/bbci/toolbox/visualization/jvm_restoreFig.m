function jvm_restoreFig(jvm, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'fig_hidden', 0);

if ~isempty(jvm) && ~opt.fig_hidden,
  set(jvm.fig, 'Visible',jvm.visible);
end
