function mrk= mrk_addPictureName(mrk, file_name, log_name)
%mrk= mrk_addPictureName(mrk, file_name, log_name)

global EEG_RAW_DIR

if log_name(1)~=filesep,
  log_name= [EEG_RAW_DIR log_name];
end
if ~ismember('.', log_name),
  log_name= [log_name '.log'];
end

[pos, toe, pic_name]= textread(log_name, '%d %d %s');

iValid= find(ismember(toe,[101 102]));
pos= pos(iValid);
toe= toe(iValid);
pic_name= pic_name(iValid);
pic_name= str_rmCommonPrefix(pic_name);

mtab= readMarkerTable(file_name, mrk.fs);
iStart= min(find(mtab.toe==251));
if ~isempty(iStart),
  start_pos= mtab.pos(iStart);
else
  iStart= min(find(mtab.toe==252));
  start_pos= mtab.pos(iStart) - 30000/1000*mtab.fs;
end
pos= pos/1000*mrk.fs + start_pos;

mrk.pic_list= unique(pic_name);
for ee= 1:length(mrk.toe),
  dd= abs(pos-mrk.trg.pos(ee));
  [mm,mi]= min(dd);
  mrk.pic(ee)= strmatch(pic_name{mi}, mrk.pic_list, 'exact');
end

mrk.indexedByEpochs= cat(2, mrk.indexedByEpochs, {'pic'});
