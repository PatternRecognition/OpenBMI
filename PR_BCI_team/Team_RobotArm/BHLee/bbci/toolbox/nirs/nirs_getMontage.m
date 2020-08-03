function mnt = nirs_getMontage(varargin)
% NIRS_GETMONTAGE - get montage with NIRS channel positions for a given set
%         of sources and detectors. NIRS channels are placed half-way 
%         between a source and a detector on a spherical head model.
%
% Synopsis:
%   MNT = nirs_getMontage(CLABSOURCE,CLABDETECTOR,<OPT>)
%         If you give channels labels of the source and detectors (should
%         correspond to EEG channel labels), montages for source and
%         detectors are produced, as well as montages for the
%         source-detector combinations.
%
%   MNT = nirs_getMontage(MNT,<OPT>)
%         You can provide a montage struct wherein source and detector 
%         montages are already given in .source and .detector fields.
%         In this case, only the source-detector combinations are determined.
%
% OPT - struct or property/value list of optional properties:
%   'file'      : montage file used for channel positions
%   'clabPolicy': specifies how the source-detector combinations are
%                 labeled. 'label' (default) concatenates the labels of
%                 source and detector (eg. Fz-Cz), 'number' concatenates
%                 the channel numbers (eg. 1-12)
%   'connector' : specify how source and detectors labels are connected
%                 (default '_'). To have no connector, set to ''.
%   'projection' : determine projection of 3D coordinates to 2D
%                  'orthogonal' or 'euclidean' (default)
%
% OUT: mnt          - montage struct of the nirs-channels
%      channel_info - cell array with nirs-channel info structs (one per channel)
%
% Remark: 
%
% Returns:
%   MNT: NIRS montage 
%        .x     - x coordinate of channel positions (2d projection)
%        .y     - y coordinate of channel positions (2d projection)
%        .pos_3d - 3d positions on spherical head model
%        .clab: channel labels
%        .angulardist: angular distances between sources and detectors on the spherical
%               model head (in radians)
%        .sd   : source and detector index to which eg the channel belongs [1 12]
%        .source : struct with montage for the sources
%        .detector : struct with montage for the detectors
%
% See also: setElectrodeMontage, nirs_reduceMontage
%
% matthias.treder@tu-berlin.de 2011

if isstruct(varargin{1})
  mnt = varargin{1};
  opt= propertylist2struct(varargin{2:end});
else
  clabSource = varargin{1};
  clabDetector = varargin{2};
  opt= propertylist2struct(varargin{3:end});
end

[opt,isdefault] = set_defaults(opt, ...
                 'clabPolicy','label',...
                 'projection','euclidean',...
                 'file','5_5', ...
                 'connector','_');

if ~exist('mnt','var')
  % Determine source and detector montages
  mnt = struct();
  mnt.source = getChannelPositions(clabSource,opt.file,'projection',opt.projection);
  mnt.detector = getChannelPositions(clabDetector,opt.file,'projection',opt.projection);
end

               
if any(isnan(mnt.source.pos_3d(:))) || any(isnan(mnt.detector.pos_3d(:)))
  warning(['Some source/detector positions are NaN. Note that the NIRS channels '...
    'involving these sources/detectors will also have NaN positions.\n '])
end

%% Determine positions of source-detector combinations
mnt.x = [];
mnt.y = [];
mnt.pos_3d = [];
mnt.clab = {};
mnt.angulardist = [];

chanIdx = 1;
for ss=1:numel(mnt.source.clab)
  for dd=1:numel(mnt.detector.clab)
    ps = mnt.source.pos_3d(:,ss);
    pd = mnt.detector.pos_3d(:,dd);
    % 3d position of new channel (half-way between source and detector)
    % Set to norm=1 so that it's located on the head surface
    newpos = (ps + pd) / norm(ps+pd);
    % Angular distance in radians = angle between the two vectors1
    costheta = dot(ps,pd);
    theta = acos(costheta);
    % Label
    if strcmp(opt.clabPolicy,'label')
      label = [mnt.source.clab{ss} opt.connector mnt.detector.clab{dd}];
    elseif strcmp(opt.clabPolicy,'number')
      label = [num2str(ss) opt.connector num2str(dd)];
    end
    % Add to struct
    mnt.pos_3d(:,chanIdx) = newpos;
    mnt.angulardist = [mnt.angulardist theta];
    mnt.clab = {mnt.clab{:} label};
    mnt.sd(chanIdx,:) = [ss,dd];
    chanIdx = chanIdx + 1;
  end
end
mnt = projectChannels3dTo2d(mnt,opt.projection);
