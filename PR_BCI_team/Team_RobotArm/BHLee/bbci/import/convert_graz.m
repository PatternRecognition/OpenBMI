global EXPORT_DIR

subs= {'f3','f5','f7'};

for is= 1:length(subs),
  subject= subs{is};
  prepareGraz;
  
  writeGenericData(epo, mrk, 0.05);

  [t,p,r]= xyz2tpr(mnt.x_3d, mnt.y_3d, mnt.z_3d);
  fid= fopen([EEG_EXPORT_DIR epo.title '.pos'], 'w');
  for ic= 1:size(cnt.x, 2),
    fprintf(fid, ['%s,%.2f,%d,%d' 13 10], epo.clab{ic}, r(ic), ...
            round(t(ic)), round(p(ic)));
  end
  fclose(fid);
end
