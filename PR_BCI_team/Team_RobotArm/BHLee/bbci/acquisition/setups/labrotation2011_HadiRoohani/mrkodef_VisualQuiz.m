function mrk= mrkodef_VisualQuiz(mrko, varargin)


%% Set default values
stimDef= {4;
          'task'};
respDef= {10,11,12;
          'music','navigation','math'};
miscDef= {1,2,3,5,6,7,8,9;
          'run_start', 'quiz_appear', 'usr_ready','rest_start','output_marked' , ...
          'crct_marked','run_end','rest_end'};

%% Build opt struct from input argument and/or default values
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef);  % trigger of target

%% Get markers according to the stimulus class definitions
mrk = mrk_defineClasses(mrko, opt.stimDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);
mrk.resp = mrk_defineClasses(mrko, opt.respDef);


%% Enhance task marker to specify the different tasks
mrk.y = mrk.resp.y;
mrk.toe = mrk.resp.toe;
mrk.className = mrk.resp.className;
