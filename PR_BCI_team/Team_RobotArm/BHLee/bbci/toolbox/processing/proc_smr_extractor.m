function [fv, opt]= proc_smr_extractor(fv, varargin)
%PROC_SMR_EXTRACTOR
%
% average across channels; subtract mean band power at valleys from
%  band power at peak, and normalize to specified range

if length(varargin)>0 & isnumeric(varargin{1}),
  opt= struct('range', varargin{1});
  varargin= varargin(2:end);
else                                         
  opt= propertylist2struct(varargin{:});     
end                                          
opt= set_defaults(opt, ...                   
                  'range', [], ...
                  'whiskerpercentiles', [25 75], ...
                  'whiskerlength', [1 1.75]);

nChansTimesFilters= size(fv.x, 2);
nTrials= size(fv.x, 3);
nChans= nChansTimesFilters/3;
X= reshape(fv.x, [nChans 3 nTrials]);

% average across all channels
X= reshape(mean(X, 1), [3 nTrials]);

% subtract mean band power at valley from band poewr at peak
X= X(1,:) - mean(X(2:3,:),1);

% determine range to which SMR amplitude values should be normalized
if isempty(opt.range),
  perc= percentiles(X, opt.whiskerpercentiles);
  opt.range= [perc(1) - opt.whiskerlength(1)*diff(perc), ...
              perc(2) + opt.whiskerlength(2)*diff(perc)];
  opt.range(1)= max(0, opt.range(1));
end

% normalize to specified range (unless specified)
X= (X-opt.range(1))/diff(opt.range);
X= max(0, X);

% put results to output structure
fv.x= X;
fv.clab= {'smr'};
