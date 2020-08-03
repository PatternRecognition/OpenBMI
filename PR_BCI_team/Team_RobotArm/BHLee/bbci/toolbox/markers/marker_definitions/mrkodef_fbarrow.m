function mrk= mrkodef_fbarrow(mrko, opt)

stimDef= {1, 2;
          opt.classes_fb{:}};
respDef= {11, 12, 21, 22, 23, 24, 25;
          ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
          ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}], ...
          'timeout', 'timeout', 'timeout'};
miscDef= {60, 70, 71, 41, 42, 51, 52;
          'free', 'rotation', 'rotation off', ...
         'touch left correct', 'touch right correct', ...
         'touch left false', 'tough right false'};
mrk_stim= mrk_defineClasses(mrko, stimDef);
mrk_resp= mrk_defineClasses(mrko, respDef);
mrk_misc= mrk_defineClasses(mrko, miscDef);

mrk_intro= mrk_chooseEvents(mrk_stim, 1:opt.adapt_trials);
mrk_stim= mrk_chooseEvents(mrk_stim, opt.adapt_trials+1:length(mrk_stim.pos));
mrk= mrk_matchStimWithResp(mrk_stim, mrk_resp, 'removevoidclasses', 0);
mrk.duration= mrk.latency;
mrk= rmfield(mrk, 'latency');
mrk.ishit= ismember(mrk.resp_toe, [11 12]);
mrk.indexedByEpochs= {'duration', 'resp_toe', 'ishit'};
mrk.misc= mrk_misc;

mrk.intro= ...
    mrk_matchStimWithResp(mrk_intro, ...
                          mrk_chooseEvents(mrk_resp, 1:opt.adapt_trials), ...
                          'removevoidclasses', 0);
mrk.intro.duration= mrk.intro.latency;
mrk.intro= rmfield(mrk.intro, 'latency');
mrk.intro.ishit= ismember(mrk.intro.resp_toe, [11 12]);
mrk.intro.indexedByEpochs= {'duration', 'resp_toe', 'ishit'};
