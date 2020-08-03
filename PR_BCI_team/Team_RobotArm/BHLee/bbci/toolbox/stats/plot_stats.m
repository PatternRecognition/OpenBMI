function h = plot_stats(dat, varargin)
% PLOT_STATS - Plot statistical data 
%
%Usage:
% h = plot_stats(dat,<OPT>)
% h = plot_stats(dat,varnames,<OPT>)
%
%Arguments:
% dat      -  data matrix. For format, see rmanova
% 'varnames'   - CELL ARRAY of one or more factors (eg {'Speller'
%              'Attention'}). The order of the factors must correspond
%              to the order of the factors in the DAT matrix.
%
% OPT - struct or property/value list of optional properties:
% plotvars  - cell array, use 'xax' 'subplot' 'line' 
% plotopt   - cell array with plotting options for each variable
% errorbar  - 'se' (standard error),'std','var',or 'none' (default 'se')
%
%Returns:
% h    -   figure handle
%
% See also rmanova 
%
% Author(s): matthias treder 2011


varnames = [];
if nargin>1 && iscell(varargin{1})
  varnames = varargin{1};
  varargin = varargin(2:end);
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'varnames', varnames, ...
                 'errorbar','se',...
                 'level',level, ...
                 'verbose',1);

               
mdat = squeeze(mean(dat,1)); % Average over subjects

if strcmp(opt.errorbar,'se')
  merr = squeeze(mean(dat,1)) / size(dat,1);
elseif strcmp(opt.errorbar,'std')
  merr = squeeze(std(dat,1));
elseif strcmp(opt.errorbar,'var')
  merr = squeeze(var(dat,1));
end
               
%% TODO !!!

