osman_dir = '/home/tensor/blanker/Daten/eegImport/osman/';

subjectNumbers = 1:9;
vecsize = 59;	% number of datasamples at each time instance
datatype = 'float32';
sizeofdata = 4; % in bytes

for is = subjectNumbers
  fileName = ['subject' num2str(is)]
  fid = fopen([osman_dir fileName '/alldata.bin'],'r','b');
  cnt.x = transpose(fread(fid, [vecsize inf], datatype));
  cnt.fs= 100;
  cnt.clab = {'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8', ...
	      'F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','Pz','POz','Oz','P7','P5','P3','P1','P2','P4','P6','P8','PO7','PO3','PO4','PO8'};
  
  cnt.title = ['osman/' fileName];
  le = load([osman_dir fileName '/lefttrain.events']);
  ri = load([osman_dir fileName '/righttrain.events']);
  te = load([osman_dir fileName '/test.events']);
  mrk.fs = cnt.fs;
  le = le(find(le(:,2)==5),1);
  ri = ri(find(ri(:,2)==6),1);
  te = te(find(te(:,2)==7),1);
  mrk.pos = transpose(cat(1,le,ri,te));
  mrk.y = zeros(2,length(mrk.pos));
  mrk.y(1,1:length(le)) = 1;
  mrk.y(2,(1:length(ri))+length(le)) = 1;
  mrk.className = {'left','right'};
  mnt = setElectrodeMontage(cnt.clab, 'osman');
  
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













