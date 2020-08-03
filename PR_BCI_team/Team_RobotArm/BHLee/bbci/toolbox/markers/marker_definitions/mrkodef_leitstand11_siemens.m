function mrk = mrkodef_leitstand11_siemens(mrko, varargin)

% mrkodef_leistand11_siemens - prepare markers for Leitstand11 Siemens
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
%   (taken from Mr. Walters mail, which was slightly ambiguous)
%
%   STIMULI:
%   1, 2, 3, 11, 21, 22, 32, 31 -- red, yellow, black, 4xgrey
%   40, 41 -- subject moves away from message screen, subject moves back
%   50, 51 -- task engagement better/worse than average
%   81-87 -- task engagement very good...bad
%   90 -- single ack
%   91 -- batch ack
%   92 -- emergency ack
%   98 -- other ack
%
%   RESPONSES:
%   1, 2 -- cl_output?
%
%
% Author(s) Bastian Venthur, 2011-02-23


stimDef = {'S  1', 'S  2', 'S  3', 'S 11', 'S 21', 'S 22', 'S 31', 'S 32', ...
           'S 40', 'S 41', ...
           'S 50', 'S 51', ...
           'S 81', 'S 82', 'S 83', 'S 84', 'S 85', 'S 86', 'S 87', ...
           'S 90', 'S 91', 'S 92', 'S 98', ...
           'R  1', 'R  2';
           'red', 'yellow', 'black', 'blue', 'grey', 'grey', 'grey', 'grey', ...
           'away', 'returning', ...
           'te-better', 'te-worse', ...
           'te1', 'te2', 'te3', 'te4', 'te5', 'te6', 'te7', ...
           'ack-one', 'ack-all', 'ack-emergency', 'ack-something-else', ...
           'cl-1', 'cl-2'};

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'stimDef', stimDef);
mrk_stim = mrk_defineClasses(mrko, opt.stimDef);
mrk = mrk_stim;

