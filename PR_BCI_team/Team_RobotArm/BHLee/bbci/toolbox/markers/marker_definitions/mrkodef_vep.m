function mrk= mrkodef_VEP(mrko, varargin)

stimDef= {{'S 10', 'S 30'};
           'stim'};
miscDef= {'S40',    'S41';
          'tba', 'tba'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);
if ~isempty(opt.miscDef)
  mrk.misc= mrk_defineClasses(mrko, opt.miscDef);
end
