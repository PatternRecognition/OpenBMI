function writeChannelPositions(mnt, file);
%writeChannelPositions(mnt, file);
%
% GLOBZ EEG_EXPORT_DIR

if file(1)==filesep,
  fullName= file;
else
  global EEG_EXPORT_DIR
  fullName= [EEG_EXPORT_DIR file];
end

[t,p,r]=  xyz2tpr(mnt.pos_3d(1,:), mnt.pos_3d(2,:), mnt.pos_3d(3,:));
fid= fopen([fullName '.pos'], 'w');
if fid==-1,
  error(sprintf('trouble opening channel position file <%s> for writing', ...
                [fullName '.pos']));
end

for ic= 1:length(t),
  fprintf(fid, ['%s, %.2f, %d, %d' 13 10], mnt.clab{ic}, r(ic), ...
          round(t(ic)), round(p(ic)));
end
fclose(fid);
