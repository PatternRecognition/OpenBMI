function h = nirs_plotOptodesAndChannels(mnt)
% NIRS_PLOTOPTODESANDCHANNELS - plots the optodes (sources and detectors)
% and the channels on a spherical head model.
%
% Synopsis:
%   MNT = nirs_plotOptodesAndChannels(MNT)

% Matthias 
% Plot detectors and sources and NIRS channels -- 3d
h = strukt();

figure
opt = {'Marker' 'o' 'MarkerEdgeColor' 'k'};
sphere(100);
axis square equal
hold all
colormap gray
if isfield(mnt,'source')
  for dd=1:numel(mnt.source.clab)
    p = mnt.source.pos_3d(:,dd);
    plot3(p(1),p(2),p(3),opt{:},'MarkerFaceColor','g');
    text(p(1),p(2),p(3),mnt.source.clab{dd},'color',[.1 .7 .1],'fontweight','bold')
  end
end
if isfield(mnt,'detector')
  for dd=1:numel(mnt.detector.clab)
    p = mnt.detector.pos_3d(:,dd);
    plot3(p(1),p(2),p(3),opt{:},'MarkerFaceColor','r');  
    text(p(1),p(2),p(3),mnt.detector.clab{dd},'color',[.7 .1 .1],'fontweight','bold')
  end
end
for dd=1:numel(mnt.clab)
  p = mnt.pos_3d(:,dd);
  plot3(p(1),p(2),p(3),opt{:},'MarkerFaceColor','b');
end
