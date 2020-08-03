function mrk= mrkodef_braking(mrko, varargin)

stimDef= {'S  2'  ,  'S  1';
          'target', 'target off'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef);

mrk = mrk_defineClasses(mrko, opt.stimDef);

