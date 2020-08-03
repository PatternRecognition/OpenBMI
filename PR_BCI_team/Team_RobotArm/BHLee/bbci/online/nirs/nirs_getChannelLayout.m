
function [channel_info, mnt] = nirs_getChannelLayout(clab, opt)
%
% IN:  clab         - consisting of clab.emit and clab.detect (electrode labels of the optode positions) 
% OUT: channel_info - cell array with nirs-channel info structs (one per channel)
%      mnt          - montage struct of the nirs-channels

if ~exist('opt','var')
    opt = [];
end
opt = set_defaults(opt, 'max_dist', 0.22);

optode_pos.emit = getElectrodePositions(clab.emit);
optode_pos.detect = getElectrodePositions(clab.detect);
d = getDistanceMatrix(optode_pos);

nEmit = length(clab.emit);
channel_info = {nEmit};
nirs_clab = {nEmit};
pos = {nEmit};
for emitIdx = 1:nEmit
   % get NIRS channels that are associated with this emitter 
   % number of NIRS channels for this emitter equals number of neighbouring detectors
   [channel_info{emitIdx}, nirs_clab{emitIdx}, pos{emitIdx}] = getNeighbouringNirsChannels(emitIdx, opt.max_dist, d, optode_pos,clab);
end

% flatten cell arrays
channel_info = cell_flaten(channel_info);
nirs_clab = cell_flaten(nirs_clab);
pos = cell_flaten(pos);
mnt = get_mnt(pos, nirs_clab);


function [channels, nirs_clab, pos] = getNeighbouringNirsChannels(emit_idx, max_dist, d, optode_pos,clab)
% find neighbouring detector indices
detect_idx = find(d(emit_idx,:)<max_dist);
% calculate resulting nirs channels
nDet = length(detect_idx);
channels = {nDet};
nirs_clab = {nDet};
pos = {nDet};
for n = 1:length(detect_idx)
    channel.x = (optode_pos.emit.x(emit_idx)+optode_pos.detect.x(detect_idx(n))) / 2;
    channel.y = (optode_pos.emit.y(emit_idx)+optode_pos.detect.y(detect_idx(n))) / 2;
    channel.emitter = emit_idx;
    channel.detector = detect_idx(n);
    channel.name = [int2str(channel.emitter) '_' int2str(channel.detector)];
    channel.EEGname=[clab.emit{channel.emitter} '_' clab.detect{channel.detector}];
    channels{n} = channel;
    nirs_clab{n} = channel.name;
    pos{n} = [channel.x channel.y];
end


function d = getDistanceMatrix(pos)
%
%   IN:     pos (with fields .emit.x/.emit.y and .detect.x/.detect.y)
%   OUT:    d  - distance matrix (pairwise between emitters and detectors)
%                d(i,j) is the distance of emitter i to detector j
%   the distance calculated is the euclidean distance between the electrode
%   positions on the 2d-scalp projection (see bbci-toolbox)

nEmit = length(pos.emit.x);
nDetect = length(pos.detect.x);
d = zeros(nEmit,nDetect);
for em = 1:nEmit
    x_em = pos.emit.x(em);
    y_em = pos.emit.y(em);
    for de = 1:nDetect
        x_de = pos.detect.x(de);
        y_de = pos.detect.y(de);
        dist = sqrt((x_em - x_de).^2+(y_em - y_de).^2);
        d(em,de) = dist;
    end
end

function mnt = get_mnt(pos, nirs_clab)
    position = cell2mat(pos);
    mnt.x = position(1:2:end); 
    mnt.y = position(2:2:end);
    mnt.clab = nirs_clab;
