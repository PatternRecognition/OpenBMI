function out= proc_laplace(dat, laplace, appendix, varargin)
%dat_lap= proc_laplace(dat, <filter_type, appendix, remainChans>)
%dat_lap= proc_laplace(dat, grid, <appendix, remainChans>)
%dat_lap= proc_laplace(dat, laplace, <appendix, remainChans>)
%
% IN   dat            - data structure of continuous or epoched data
%      filter_type    - {small, large, horizontal, vertical, diagonal, eight},
%                       default 'small'
%      grid           - 2-d cell array of channel labels (set getGrid)
%      laplace
%             .grid   - 2-d cell array of channel labels (set getGrid)
%             .filter - spatial filter, e.g., [0 -1; 1 0; 0 1; -1 0]
%      appendix       - is appended to channel labels, default ' lap'
%      remainChans    - copy those channels, default []
%
% OUT  dat_lap        - updated data structure
%
% if 'remainChans' is set to 'filter all', then every channel is
% laplace-filtered, even those having incomplete (or no) neighbours
%
% to plot scalp maps of laplace filtered signals you may need to define
%   mnt_lap= restrictDisplayMontage(mnt, epo_lap);
%
% if you have channel labels according to the extended international
% 10/10 system and you want to apply the standard laplace filter just use
%   dat= proc_laplace(dat)
%
% SEE  getGrid, restrictDisplayMontage

% bb, ida.first.fhg.de


if ~exist('laplace','var') | isempty(laplace), laplace= 'small'; end
if ~exist('appendix','var'), appendix= ' lap'; end
if length(varargin)==1 & strcmpi(varargin{1}, 'filter all'),
  FILTER_ALL= 1;
else
  FILTER_ALL= 0;
end

  
if isstruct(laplace),
  if ~isfield(laplace, 'filter'), laplace.filter='small'; end
  if ischar(laplace.filter),
    laplace.filter= getLaplaceFilter(laplace.filter);
  end
else
  if iscell(laplace),
    laplace.grid= laplace;
    laplace.filter= [0 -1; 2 0; 0 1; -2 0]';
  else
    filter_type= laplace;
    laplace.grid= getGrid('visible_128');
    laplace.filter= getLaplaceFilter(filter_type);
  end
end


nChans= length(dat.clab);
pos= zeros(2, nChans);
for ic= 1:nChans,
  pos(:,ic)= getCoordinates(dat.clab{ic}, laplace.grid);
end

nRefs= size(laplace.filter,2);
lc= 0;
A= zeros(nChans, 0);
for ic= 1:nChans,
  refChans= [];
  for ir= 1:nRefs,
    ri= find( pos(1,:)==pos(1,ic)+laplace.filter(1,ir) & ...
              pos(2,:)==pos(2,ic)+laplace.filter(2,ir) );
    refChans= [refChans ri];
  end
  if length(refChans)==nRefs | FILTER_ALL,
    lc= lc+1;
    A(ic,lc)= 1;
    if ~isempty(refChans),
      A(refChans,lc)= -1/length(refChans);
    end
  end
end
A= [A; zeros(nChans-size(A,1), size(A,2))];

out= proc_linearDerivation(dat, A, appendix);
if ~FILTER_ALL,
  out= proc_copyChannels(out, dat, varargin);
end



function pos= getCoordinates(lab, grid)

nRows= size(grid,1);
%w_cm= warning('query', 'bci:missing_channels');
%warning('off', 'bci:missing_channels');
ii= chanind(grid, lab);
%warning(w_cm);
if isempty(ii),
  pos= [NaN; NaN];
else    
  xc= 1+floor((ii-1)/nRows);
  yc= ii-(xc-1)*nRows;
  xc= 2*xc - isequal(grid{yc,1},'<');
  pos= [xc; yc];
end



function filt= getLaplaceFilter(filter_type)

switch lower(filter_type),
 case 'small',
  filt= [0 -2; 2 0; 0 2; -2 0]';
 case 'large',
  filt= [0 -4; 4 0; 0 4; -4 0]';
 case 'horizontal',
  filt= [-2 0; 2 0]';
 case 'vertical',
  filt= [0 -2; 0 2]';
 case 'diagonal',
  filt= [-2 -2; 2 -2; 2 2; -2 2]';
 case 'eight',
  filt= [-2 0; -1 -1; 0 -2; 1 -1; 2 0; 1 1; 0 2; -1 1]';
 otherwise
  error('unknown filter matrix');
end
