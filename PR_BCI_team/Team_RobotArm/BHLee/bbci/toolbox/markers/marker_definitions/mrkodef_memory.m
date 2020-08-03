function mrk= mrkodef_memory(mrko, opt)

%     stimDef= {102,202,103,203,171,172,150,160,151,152,161,162,111; ...
%         'show','ask','inter_period','inter_period_test','correct','incorrect','testing','training','test_corr', ...
%         'test_incorr','train_corr','train_incorr','possible_learn'};
    % New definitions one show training paradigm
    stimDef= {102,202,103,203,171,172,150,160,151,152,161,162,111,30,31,32,115,116; ...
        'show','ask','inter_period','inter_period_test','correct','incorrect','testing','training','test_corr', ...
        'test_incorr','train_corr','train_incorr','possible_learn','new_pair','old_pair','Test','first_press', ...
        'last_press'};

    %espDef= {171,172;'correct','incorrect'};
    %miscDef= {60; 'intraining'};
    mrk_stim= mrk_defineClasses(mrko, stimDef);
    %rk_resp= mrk_defineClasses(mrko, respDef);
    %mrk_misc= mrk_defineClasses(mrko, miscDef);

    %rk= mrk_matchStimWithResp(mrk_stim, mrk_resp, 'removevoidclasses', 0);
    %rk.duration= mrk.latency;
    %rk= rmfield(mrk, 'latency');
    mrk=mrk_stim;