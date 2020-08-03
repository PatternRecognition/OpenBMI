function mrk = mrkodef_leitstand10_triggertest(mrko, varargin)

% mrkodef_leistand10_triggertest - prepare markers for Leitstand10 triggertest
% evaluation.
%
% Synopsis:
%   mrk = mrkodef_leitstand10_triggertest(mrko, varargin)
%
% Arguments:
%   mrko - ???
%   varargin - ???
%
% Output:
%   mrk - ???
%
% Marker codings:
%
%
% Author(s) Bastian Venthur, 2010-11-17


% TODO: add correct stimuly
stimDef = {'R  4', 'R  5', 'S  1', 'S128'; 'optic_red', 'optic_yellow', 'red', 'yellow'};

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'stimDef', stimDef);
mrk_stim = mrk_defineClasses(mrko, opt.stimDef);
mrk = mrk_stim;

