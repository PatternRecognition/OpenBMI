function mrk= mrkodef_imag_sssep(mrko, varargin)

stimDef= {'S  1', 'S  2',  'S  3';
          'Left', 'Right', 'Foot'};
miscDef= {'S  5', 'S  6'; 
          'Start', 'End'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);
