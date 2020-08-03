setup_augcog;
fid= 1;

nn= 1;

mrk_cmt= readMarkerComments(augcog(nn).file, 100);

fprintf(fid, 'file: %s\n', augcog.file{nn});
for ii= 1:length(mrk_cmt.pos),
  sec_tot= round(mrk_cmt.pos(ii)/100);
  mn= floor(sec_tot/60);
  sc= sec_tot-60*mn;
  fprintf(fid, '%3d''%02d: %s\n', mn, sc, mrk_cmt.str{ii});
end
if fid>2,
  fclose(fid);
end
