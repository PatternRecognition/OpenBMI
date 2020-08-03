function mnt = getChannelPositions(clab, varargin)
%GETCHANNELPOSITIONS - returns montage with channel positions of channels
%                      defined in a montage file
%
%Usage:
% mnt = getChannelPositions(clab,<OPT>)
% mnt = getChannelPositions(clab,file,<OPT>)
%
%Arguments:
% CLAB  - channel labels
% OPT - struct or property/value list of optional properties:
% 'file' -        file with channel montage (default '10_10'). Standard
%                 montage files are located in
%                 toolbox/visualization/montages/. If you want to
%                 specify a different location, an absolute path should be
%                 given.
% 'projection' -  the method for projecting 3D electrode positions onto 2D.
%                 See project3dChannelPositions.
%
%Returns:
% MNT:  Struct for electrode montage
%   .pos_3d - 3d positions on spherical head model
%   .x     - x coordinate for 2D projection
%   .y     - y coordinate for 2D projection
%   .clab  - channel labels
%
% The 3D positioning assumes a spherical head model with radius 1.
%
% See also projectChannels3dTo2d

global BCI_DIR

if mod(nargin,2)==0
  varargin = {'file' varargin{:}};
end
  
opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'projection','sphere', ...
                 'file','10_10');

               
%% Read montage file
fullName = fullfile([BCI_DIR 'toolbox/visualization/montages/'],opt.file);
if ~exist(fullName,'file'), fullName = [fullName '.txt']; end

fid= fopen(fullName, 'r');
if fid==-1, error(sprintf('%s not found', fullName)); end

pat = '%s %n %n %n';
r = textscan(fid,pat,'commentstyle','#');
[allclab x y z] = deal(r{:});

%% Set mnt struct
idx = chanind(allclab,clab);
mnt = [];
mnt.clab = allclab(idx);
mnt.pos_3d = [x(idx)'; y(idx)'; z(idx)'];

%% Project to 2D
mnt = projectChannels3dTo2d(mnt,opt.projection);

