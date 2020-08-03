function clab= getClabForLAR(dat, mnt, tbf, varargin)
% GETCLABFORLAR - Get channels which are required for LAR filtering
%
%Synopsis:
%  CLAB= getClabForLAR(DAT, MNT, TBF, RADIUS)
%  CLAB= getClabForLAR(DAT, MNT, TBF, <OPT>)
%
%Arguments:
%  DAT: Data structure of continuous or epoched data
%  MNT: Electrode montage, see getElectrodePositions
%  TBF: Channels that should be filtered by LAR
%  RADIUS: see OPT.radius
%  OPT: struct or property/value list of optional properties:
%    .radius: For a radius of 0.8, the neighborhood of Cz extends from
%      C3 to C4 and from Fz to Pz. For a radius of 0.4 it extends from
%      C1 to C2 and from FCz to CPz. Default 0.8.
%    .clab: Labels of channels which are considered (and rereferenced).
%    .median: If true, reference is calculated as median across
%      neighboring channels, instead of mean. Default 0.
%    .verbose: If true, text output shows the rereferencing
%
%Returns:
%  CLAB: Labels of the channels that are needed for LAR in order
%     to obtain the channels specified by TBF.
%
%See:
%  proc_localAverageReference

% Author: Benjamin Blankertz

if length(varargin)==1,
  opt= struct('radius', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'radius', 0.8, ...
                  'clab', {'not','E*'});

mnt= mnt_adaptMontage(mnt, dat);
if ~isequal(mnt.clab, dat.clab),
  error('channel mismatch');
end

requ_clab= {};
rc= chanind(dat, opt.clab);
idx_tbf= chanind(dat, tbf);
for cc= idx_tbf,
  pos= repmat(mnt.pos_3d(:,cc), [1 length(rc)]);
  dist= sqrt(sum( (mnt.pos_3d(:,rc)-pos).^2) );
  iRef= find(dist<opt.radius);
  requ_clab= unique(cat(2, requ_clab, dat.clab{rc(iRef)}));
end

%% This maintains the order of the channel labels in dat.clab
idx= strpatternmatch(requ_clab, dat.clab);
clab= dat.clab(idx);
