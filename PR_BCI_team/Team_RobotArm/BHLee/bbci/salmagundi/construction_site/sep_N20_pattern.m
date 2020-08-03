fs= 1000;

exp_sub= 'Gabriel';
exp_date= '01_12_12';
type_list= {'medianus_left', 'medianus_right'};

ival= [-20 250];
nFiles= length(type_list);
colormap('default');
col= colormap;
col= col(4:4:64,:);
colormap(col);
for fi= 1:nFiles,
  subplot(1, nFiles, fi);
  file= [exp_sub '_' exp_date '/' type_list{fi} exp_sub];
  cnt= readGenericEEG(file, [], fs);
  mrk= readMarkerTable(file, fs);
  stim= find(mrk.toe==32);
  mrk= pickEvents(mrk, stim);
  epo= makeSegments(cnt, mrk, ival);
  mnt= setElectrodeMontage(epo.clab);
  sep= proc_classMean(epo);
  sep= proc_baseline(sep, [16 18]);
  ti= getIvalIndices([21 22], sep);
  w= mean( sep.x(ti, :), 1 );
  showScalpPattern(mnt, w, 0, 'horiz');
  title(untex(type_list{fi}));
end
