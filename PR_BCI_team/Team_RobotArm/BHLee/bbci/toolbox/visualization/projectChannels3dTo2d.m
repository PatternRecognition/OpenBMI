function mnt = projectChannels3dTo2d(mnt,varargin)

%PROJECTCHANNELS3DTO2D - projects 3D channel positions in .pos_3d to a
%                            2D map using different projection methods.
%
%Usage:
% mnt = projectChannels3dTo2d(mnt)
% mnt = projectChannels3dTo2d(mnt,projection)
%
%Arguments:
% MNT - montage struct with 3D positions
% PROJECTION -  the method for projecting 3D electrode positions onto 2D.
%               'euclidean': euclidean distance from the channel
%                to the upper vertex ('Cz')
%               'orthogonal': orthogonal projection of the upper half-sphere
%               'sphere': similar to euclidean but takes the distance on the
%                 sphere instead of the Euclidean distance (default
%                 'sphere')
%
%Returns:
% MNT:  Struct for electrode montage
%   .x     - x coordinate for 2D projection
%   .y     - y coordinate for 2D projection
%
% See also getChannelPositions

if mod(nargin,2)==0
  varargin = {'projection' varargin{:}};
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'projection','sphere');

               
%% Projection to 2D
switch(opt.projection)
  
  case 'orthogonal'
  % only positive z-values are considered [upper half-sphere]
  mnt.x = mnt.pos_3d(1,:)';
  mnt.y = mnt.pos_3d(2,:)';
  negz = mnt.pos_3d(3,:)<0;
  mnt.x(negz) = nan;
  mnt.y(negz) = nan;
  
  case 'euclidean'
  % Euclidean projection with projection center middle-top of sphere
  % (approx. Cz). The distances to the channels correspond to the Euclidean
  % distance between (0/0/1) and the channel in 3D.
  projectionCenter = [0 0 1];
  % Euclidean distance to projection center
  dist = mnt.pos_3d - repmat(projectionCenter',[1 length(mnt.pos_3d)]);
  dist = sqrt(sum(dist .^ 2));
  % The distance from [0 0 1] to the inion [0 -1 0] is sqrt(2) but in the
  % projection it should be one, therefore divide by sqrt 2 also
  dist = dist ./ sqrt(2);
  % Set the length of the 2D vectors equal to dist
  xy  = mnt.pos_3d(1:2,:);
  twoDnorms = sqrt(sum(xy.^ 2));
  twoDnorms(twoDnorms==0)=1;   % to prevent division by 0 next
  xy = xy ./ repmat(twoDnorms,[2 1]) .* repmat(dist,[2 1]);
  mnt.x = xy(1,:)';
  mnt.y = xy(2,:)';

  case 'sphere'
  x=mnt.pos_3d(1,:);
  y=mnt.pos_3d(2,:);
  z=mnt.pos_3d(3,:);
  % This projection uses the distance on the "head"-surface to determine the 2d-positions of the electrodes w.r.t. Cz.
  alpha = asin(sqrt(x.^2 + y.^2));
  stretch = 2-2*abs(alpha)/pi;
  stretch(z>0) = 2*abs(alpha(z>0))/pi;
  norm = sqrt(x.^2 + y.^2);
  norm(norm==0) = 1;
  cx = x.*stretch./norm;
  cy = y.*stretch./norm;
  mnt.x = cx;
  mnt.y = cy;

end