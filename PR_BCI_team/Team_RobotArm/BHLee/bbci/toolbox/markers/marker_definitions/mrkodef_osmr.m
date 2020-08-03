function mrk= mrkodef_osmr(mrko, opt)

classDef= {'S  1', 'S  2',  'S  3';
           'left', 'right', 'foot'};
miscDef= {'S100',    'S101',  'S249',        'S250',      'S252',  'S253';
          'cue off', 'cross', 'pause start', 'pause end', 'start', 'end'};

mrk= mrk_defineClasses(mrko, classDef);
mrk.pos= mrk.pos - 8/1000*mrk.fs;
mrk.misc= mrk_defineClasses(mrko, miscDef);
