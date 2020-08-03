function mnt= mnt_scalpToGrid(mnt, varargin)
%MNT_SCALPTOGRID - Montage for grid plot with boxes at scalp locations
%
%Usage:
%  MNT= mnt_scalpToGrid(MNT, <OPTS>)
%
%Input:
%  MNT: Display montage
%  OPTS: property/value list or struct of optional properties:
%   .clab     - choose only locations for those specified channels,
%               cell array, for format see function chanind.
%   .axisSize - [width height]: size of axis. Default [] means choosing
%               automatically the large possible size, without overlapping.
%   .oversize - factor to increase axisSize to allow partial overlapping,
%               default 1.2.
%   .maximize_angle - when choosing automatically the axisSize, the
%               criterium is to maximize size in direction of this angle.
%   .legend_pos - [hpos vpos], where hpos=0 means leftmost, and hpos=1 means
%               rightmost edge, and vpos=0 means lower and vpos=1 means
%               upper edge.
%   .scale_pos - [hpos vpos], analog to .legend_pos
%   .pos_correction - type of corrections for channel positions. There
%               are some popular variants hard coded here. Default 0.
%
%Output:
%  MNT: Updated display montage
%
%See also setDisplayMontage, projectElectrodePositions, grid_plot,
%  mnt_restrictMontage

%% blanker@first.fhg.de 01/2005


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'axisSize', [], ...
                  'clab', [], ...
                  'oversize', 1.2, ...
                  'maximize_angle', 60, ...
                  'legend_pos', [0 0], ...
                  'scale_pos', [1 0], ...
                  'pos_correction', 0);

if length(opt.oversize)==1,
  opt.oversize= opt.oversize*[1 1];
end

chind= find(~isnan(mnt.x));
if ~isempty(opt.clab),
  chind= setdiff(chind, chanind(mnt, opt.clab));
end

mnt.box= NaN*zeros(2,length(mnt.clab)+1);
mnt.box(:,chind)= [mnt.x(chind)'; mnt.y(chind)'];

if isempty(opt.axisSize) | isequal(opt.axisSize, 'auto'),
  min_dx= inf;
  min_dy= inf;
  for ii= 1:length(chind)-1,
    for jj= ii+1:length(chind),
      mini= min(mnt.box(:,chind([ii jj])), [], 2);
      maxi= max(mnt.box(:,chind([ii jj])), [], 2);
      dx= maxi(1)-mini(1);
      dy= maxi(2)-mini(2);
      ang= 180/pi* atan2(dx, dy);
      if ang<opt.maximize_angle,
        if dy<min_dy,
          min_dy= dy;
        end
      else
        if dx<min_dx,
          min_dx= dx;
        end
      end
    end
  end
  opt.axisSize= [min_dx min_dy];
end

mnt.box_sz= diag(opt.oversize)*opt.axisSize(:)*ones(1,size(mnt.box,2));

mi_x= min(mnt.x(chind));
ma_x= max(mnt.x(chind));
mi_y= min(mnt.y(chind));
ma_y= max(mnt.y(chind));
if size(mnt.box,2)>length(mnt.clab),
  mnt.box(:,end)= [mi_x*(1-opt.legend_pos(1))+ma_x*opt.legend_pos(1); ...
                   mi_y*(1-opt.legend_pos(2))+ma_y*opt.legend_pos(2)];
end

if isfield(mnt, 'scale_box'),
  mnt.scale_box= [mi_x*(1-opt.scale_pos(1))+ma_x*opt.scale_pos(1); ...
                  mi_y*(1-opt.scale_pos(2))+ma_y*opt.scale_pos(2)];
  mnt.scale_box_sz= diag(opt.oversize)*opt.axisSize(:);
end

switch(opt.pos_correction),
 case 1,
  ci= chanind(mnt, 'AF3,4');
  mnt.box(:,ci)= [-0.2 0.2; 0.75 0.75];
  ci= chanind(mnt, 'PO7,8');
  mnt.box(:,ci)= [-0.45 0.45; -0.65 -0.65];
  ci= chanind(mnt, 'TP7,8');
  mnt.box(:,ci)= [-0.68 0.68; -0.37 -0.37];
end
