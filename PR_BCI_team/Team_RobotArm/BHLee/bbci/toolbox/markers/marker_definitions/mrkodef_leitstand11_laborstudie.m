function mrk = mrkodef_leitstand10_siemens(mrko, varargin)

% mrkodef_leistand10_siemens - prepare markers for Leitstand10 Siemens
%
% Synopsis:
%   mrk = mrkodef_leitstand11_laborstudie(mrko, varargin)
%
% Arguments:
%   mrko - ???
%   varargin - ???
%
% Output:
%   mrk - ???
%
% Marker codings:
%   20, 21, 22 -- Stimuli: blue, yellow, red
%   141, 142 -- Correct responses: yellow, red
%   241, 242 -- Incorrect responses: yellow, red
%   50, 150 -- start-, stop-Nebenaufgabe
%   80, 180 -- start-, stop-copyspeller
%
% Author(s) Bastian Venthur, 2011-01-25


stimDef = {'S 20', 'S 21', 'S 22', ...
           'R141', 'R142', 'R241', 'R242', ...
           'S 50', 'S150', 'S 80', 'S180'; ...
           'blue', 'yellow', 'red', ...
           'yellow_correct', 'red_correct', ...
           'yellow_incorrect', 'red_incorrect', ...
           'start_na', 'stop_na', ...
           'start_cs', 'stop_cs'};

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'stimDef', stimDef);
mrk_stim = mrk_defineClasses(mrko, opt.stimDef);
mrk = mrk_stim;

