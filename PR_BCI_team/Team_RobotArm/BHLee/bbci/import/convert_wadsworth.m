wolpaw_dir= '/home/tensor/blanker/Daten/eegImport/wadsworth/';

subjectList= {'AA', 'BB', 'CC'};
for is= 1:length(subjectList),
 for iz= 1:10,
  fileName= sprintf('%s%03d', subjectList{is}, iz)
  load([wolpaw_dir fileName]);
  
  cnt.x= signal;
  cnt.fs= 160;
  cnt.clab= {'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
             'C5','C3','C1','Cz','C2','C4','C6', ...
             'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
             'Fp1','Fpz','Fp2', ...
             'AF7','AF3','AFz','AF4','AF8', ...
             'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
             'FT7','FT8', 'T7','T8', 'T9','T10', ...
             'TP7','TP8', ...
             'P7','P5','P3','P1','Pz','P2','P4','P6','P8', ...
             'PO7','PO3','POz','PO4','PO8', ...
             'O1','Oz','O2', ...
             'Iz'};
  cnt.title= ['wadsworth/' fileName];

  trial_start= [1;  1+find(diff(trial)==1)]';
  mrk.pos= trial_start(1:end-1);
  mrk.fs= cnt.fs;
  trial_target= TargetCode(trial_start(2:end)-1)';
  mrk.y= [trial_target==1; trial_target==2; trial_target==3; trial_target==4];
  mrk.className= {'1', '2', '3', '4'};

  mnt= setElectrodeMontage(cnt.clab, 'small');

  saveProcessedEEG(cnt.title, cnt, mrk, mnt);
  
  writeGenericData(cnt, mrk, 16);
  
  [t,p,r]= xyz2tpr(mnt.x_3d, mnt.y_3d, mnt.z_3d);
  fid= fopen([EEG_EXPORT_DIR cnt.title '.pos'], 'w');
  for ic= 1:size(cnt.x, 2),
    fprintf(fid, ['%s,%.2f,%d,%d' 13 10], cnt.clab{ic}, r(ic), ...
            round(t(ic)), round(p(ic)));
  end
  fclose(fid);  
 end
end









