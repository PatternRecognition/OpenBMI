function mrk = mrkodef_leitstand10_siemens(mrko, varargin)

% mrkodef_leistand10_siemens - prepare markers for Leitstand10 Siemens
%
% Synopsis:
%   mrk = mrkodef_leitstand10_siemens(mrko, varargin)
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
% Author(s) Bastian Venthur, 2010-09-10


% TODO: add correct stimuly
stimDef = {'R 12', 'R 14', 'R 15'; 'white', 'red', 'yellow'};

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'stimDef', stimDef);
mrk_stim = mrk_defineClasses(mrko, opt.stimDef);
mrk = mrk_stim;

