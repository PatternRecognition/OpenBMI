function mrk= mrkdef_fixed_seq_audi(mrko)
classDef = {[11:16], [1:6],[31:36],[21:26];'targetRand','non-targetRand' ,'targetFix', 'non-targetFix'};
mrk = mrk_defineClasses(mrko, classDef);