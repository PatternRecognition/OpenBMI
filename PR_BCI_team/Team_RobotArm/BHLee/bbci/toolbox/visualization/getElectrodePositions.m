function mnt= getElectrodePositions(clab, varargin)
%GETELECTRODEPOSITIONS - Electrode positions of standard named channels
%
%Usage:
% MNT= getElectrodePositions(CLAB);
%
%Input:
% CLAB: Label of channels (according to the extended international
%       10-10 system, see calc_pos_ext_10_10).
%Output:
% MNT:  Struct for electrode montage
%   .x     - x coordiante of electrode positions
%   .y     - y coordinate of electrode positions
%   .clab  - channel labels
%
%See also mnt_setGrid

%      posSystem      - name such that ['calc_pos_ ' posSystem] is an m-file
%                       or a struct with fields x, y, z, clab
%

% kraulem 08.09.2003

if ~exist('clab','var'),
  [d,d,d,clab]= calc_pos_ext_10_10;
end
if nargin<=1 | isempty(varargin{1}),
  varargin{1}= calc_pos_ext_10_10;
end

if ischar(varargin{1}),
  posFile= ['calc_pos_' varargin{1}];
  if exist(posFile, 'file'),
    posSystem= feval(posFile);
    displayMontage= {varargin{2:end}};
  else
    posSystem= calc_pos_ext_10_10;
    displayMontage= {varargin{:}};
  end
  x= posSystem.x;
  y= posSystem.y;
  z= posSystem.z;
  elab= posSystem.clab;
elseif isstruct(varargin{1}),
  posSystem= varargin{1};
  displayMontage= {varargin{2:end}};
  x= posSystem.x;
  y= posSystem.y;
  z= posSystem.z;
  elab= posSystem.clab;
elseif nargin==5 | (nargin==4 & ~ischar(varargin{3})),
  elab= clab;
  [x,y,z]= deal(varargin{1:3});
  displayMontage= {varargin{4:end}};
else
  elab= clab;
  [x,y,z]= abr2xyz(varargin{1:2});
  displayMontage= {varargin{3:end}};
end

maz= max(z(:));
miz= min(z(:));
% ALTERNATIVE PROJECTION:
% This function works with an input of mnt which assumes that the human head "is" a ball with radius 1.
% The lower section of this ball is first projected onto the walls of a cylinder (orthogonal, with radius 1);
% then all (new) points will be projected on the 2d-hyperspace with z=maz.
%ur= [0 0 miz-0.8*(maz-miz)];
% ur is the center of projection. This is why the function only works with radius = 1 
% and ur(1:2) = [0 0].
ur= [0 0 -1.5];
if 1==0% old projection. uses center of projection.
  la= (maz-ur(3)) ./ (z(:)-ur(3));
  Ur= ones(length(z(:)),1)*ur;
  % Project the lower halfball onto the wall of the cylinder:
  cx = x;
  cy = y;
  cx(z<0) = cx(z<0)./sqrt(cx(z<0).^2+cy(z<0).^2);
  cy(z<0) = cy(z<0)./sqrt(cx(z<0).^2+cy(z<0).^2);
  % Project everything onto the plane {z = maz}:
  pos2d= Ur + (la*ones(1,3)) .* ([cx(:) cy(:) z(:)] - Ur);
  pos2d= pos2d(:, 1:2);
  %pos2d(z<0,:)= NaN;% TODO: don't throw away the values of the lower halfball!
end

% This projection uses the distance on the "head"-surface to determine the 2d-positions of the electrodes w.r.t. Cz.
alpha = asin(sqrt(x.^2 + y.^2));
stretch = 2-2*abs(alpha)/pi;
stretch(z>0) = 2*abs(alpha(z>0))/pi;
norm = sqrt(x.^2 + y.^2);
norm(norm==0) = 1;
cx = x.*stretch./norm;
cy = y.*stretch./norm;
pos2d = [cx(:) cy(:)];

nChans= length(clab);
mnt.x= NaN*ones(nChans, 1);
mnt.y= NaN*ones(nChans, 1);
mnt.pos_3d= NaN*ones(3, nChans);
for ei= 1:nChans,
  ii= chanind(elab, clab{ei});
  if ~isempty(ii),
    mnt.x(ei)= pos2d(ii, 1);
    mnt.y(ei)= pos2d(ii, 2);
    mnt.pos_3d(:,ei)= [x(ii) y(ii) z(ii)];
  end
end
radius = 1.3;
%radius= 1.9;
mnt.x= mnt.x/radius;
mnt.y= mnt.y/radius;
mnt.clab= clab;

if ~isempty(displayMontage),
  mnt= mnt_setGrid(mnt, displayMontage{:});
end
