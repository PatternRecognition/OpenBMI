function [fv, opt]= proc_smr_extractor_adaptCentralBand(fv, varargin)
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
                  'whiskerlength', [1 1.75], ...
                  'normalize_within_bands', 0, ...
                  'scale', [], ...
                  'substract_band_numb', 2);


              
nChansTimesFilters= size(fv.x, 2);
nTrials= size(fv.x, 3);
nChans= nChansTimesFilters/(opt.substract_band_numb+1);
%nChans= nChansTimesFilters/2;
X= reshape(fv.x, [nChans opt.substract_band_numb+1 nTrials]);
%X= reshape(fv.x, [nChans 2 nTrials]);

% average across all channels
X= reshape(mean(X, 1), [opt.substract_band_numb+1 nTrials]);
%X= reshape(mean(X, 1), [2 nTrials]);

if opt.normalize_within_bands & isempty(opt.scale),
  opt.scale= abs(1./mean(X,2));
end

if opt.normalize_within_bands,
  X= X .* repmat(opt.scale, [1 nTrials]);
end

% subtract mean band power at valley from band power at peak
%X= X(1,:) - X(2,:);
X= X(1,:) - mean(X(2:(opt.substract_band_numb+1),:),1);

% determine range to which SMR amplitude values should be normalized
if isempty(opt.range),
  perc= percentiles(X, opt.whiskerpercentiles);
  opt.range= [perc(1) - opt.whiskerlength(1)*diff(perc), ...
              perc(2) + opt.whiskerlength(2)*diff(perc)];
  %opt.range(1)= max(0, opt.range(1));
end

% normalize to specified range (unless specified)
X= (X-opt.range(1))/diff(opt.range);
%X= max(0, X);
X= max(min(0,opt.range(1)),X);

% put results to output structure
fv.x= X;
fv.clab= {'smr'};
