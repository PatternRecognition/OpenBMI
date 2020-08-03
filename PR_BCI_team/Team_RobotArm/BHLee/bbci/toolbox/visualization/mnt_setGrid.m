function mnt= mnt_setGrid(mnt, displayMontage, varargin)
%MNT_SETGRID - Define a new eletrode grid layout for an electrode montage
%
%Usage:
% mnt= mnt_setGrid(mnt, displayMontage, <opts>)
%
%Input:
% mnt            - struct for electrode montage, see setElectrodeMontage
% displayMontage - any *.mnt file in EEG_CFG_DIR, 
%                  e.g., 'small', 'medium', 'large',
%                  or a string defining the montage (see example)
% opts - struct or property/value list of optional field:
%   .centerClab  - label of channel to be positioned at (0,0)
%
%Output:
% mnt: updated struct for electrode montage
%
%Example:
% grd= sprintf('legend,Fz,scale\n,C3,Cz,C4\nP3,Pz,P4');
% mnt= mnt_setGrid(mnt, grd);
%
%See also: getElectrodePositions, getGrid.

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
		  'centerClab', 'Cz');

if ~exist('displayMontage', 'var'), displayMontage='medium'; end
if ~isstruct(mnt),
  mnt= struct('clab', {mnt});
end

grid= getGrid(displayMontage);
if ~any(ismember(strhead(mnt.clab), grid)),
  return;
end

%w_cm= warning('query', 'bci:missing_channels');
%warning('off', 'bci:missing_channels');
clab= cat(2, strhead(mnt.clab), {'legend','scale'});
nChans= length(clab);
mnt.box= zeros(2, nChans);
mnt.box_sz= ones(2, nChans);
[c0,r0]= getIndices(opt.centerClab, grid);
if isnan(c0), 
  c0=0; r0=0; 
end
for ei= 1:nChans,
  [ci,ri]= getIndices(clab{ei}, grid);
  if length(ci)>1,
    warning('channel %s appears multiple times in the grid layout', clab{ei});
  end
  if isnan(ri),
    mnt.box(:,ei)= [NaN; NaN];
  else
    if isequal(grid{ri,1},'<'),
      cc= c0+0.5;
    else
      cc= c0;
    end
    for ii= 1:length(ci),  %% loop is needed for the case that one channel appear multiple times in the grid
      mnt.box(:,ei)= [ci(ii)-cc; -(ri(ii)-r0)];
    end
  end
end
%warning(w_cm);
mnt.scale_box= mnt.box(:,end);
mnt.scale_box_sz= mnt.box_sz(:,end);
mnt.box= mnt.box(:,1:end-1);
mnt.box_sz= mnt.box_sz(:,1:end-1);



function [ci,ri]= getIndices(lab, grid)

nRows= size(grid,1);
ii= chanind(grid, lab);
if isempty(ii),
  ci= NaN;
  ri= NaN;
else    
  ci= 1+floor((ii-1)/nRows);
  ri= ii-(ci-1)*nRows;
end
