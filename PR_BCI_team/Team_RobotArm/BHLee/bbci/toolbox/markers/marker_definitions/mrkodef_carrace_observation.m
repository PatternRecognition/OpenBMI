function mrk= mrkdef_carrace_drive(mrko, varargin)

stimDef= {'S  4', 'S  8', 'S 16', 'S 32';
          'car_normal', 'car_brake', 'car_hold', 'car_collision'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef);

mrk = mrk_defineClasses(mrko, opt.stimDef);

