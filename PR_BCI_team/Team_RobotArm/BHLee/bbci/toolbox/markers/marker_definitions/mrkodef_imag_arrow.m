function mrk= mrkodef_imag_arrow(mrko, varargin)
%% pseudo online
% stimDef= {1, 2, 3;
%           'left','right', 'foot'};
% stimDef= {'1','S 11','S  2';
%            'start','Walking','Resting'};
% stimDef= {'S  1', 'S  2',  'S  3', 'S  4', 'S  5';
%            'test', 'walking start', 'walking stop','rest start', 'rest stop'};
%  miscDef= {'S100',    'S101',  'S249',        'S250',      'S252',  'S253';
%            'cue off', 'cross', 'pause start', 'pause end', 'start', 'end'};
% 
% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt, 'stimDef', stimDef);
% 
%                    
% %                    opt= set_defaults(opt, 'stimDef', stimDef, ...
% %                        'miscDef', miscDef);
% 
% mrk= mrk_defineClasses(mrko, opt.stimDef);
% mrk.misc= mrk_defineClasses(mrko, opt.miscDef);


%% walking
stimDef= {'1','S 11';
           'start','Walking'};
% stimDef= {'S  1', 'S  2',  'S  3', 'S  4', 'S  5';
%            'test', 'walking start', 'walking stop','rest start', 'rest stop'};
 miscDef= {'S100',    'S101',  'S249',        'S250',      'S252',  'S253';
           'cue off', 'cross', 'pause start', 'pause end', 'start', 'end'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef);

                   
%                    opt= set_defaults(opt, 'stimDef', stimDef, ...
%                        'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);

%% resting
% stimDef= {'1','S  2';
%            'start','Resting'};
% % stimDef= {'S  1', 'S  2',  'S  3', 'S  4', 'S  5';
% %            'test', 'walking start', 'walking stop','rest start', 'rest stop'};
%  miscDef= {'S100',    'S101',  'S249',        'S250',      'S252',  'S253';
%            'cue off', 'cross', 'pause start', 'pause end', 'start', 'end'};
% 
% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt, 'stimDef', stimDef);
% 
%                    
% %                    opt= set_defaults(opt, 'stimDef', stimDef, ...
% %                        'miscDef', miscDef);
% 
% mrk= mrk_defineClasses(mrko, opt.stimDef);