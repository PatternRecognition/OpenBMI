%MAT_DIR= '/home/data/BCI/bbciMat/';
MAT_DIR= '/mnt/usb/data/bbciMat/';
OUT_DIR= '/mnt/usb/tmp/';

subdir_list= textread([BCI_DIR 'investigation/studies/season2/session_list_ext'], '%s');

for vp= 1:length(subdir_list),

subdir= [subdir_list{vp} '/'];
mkdir([OUT_DIR subdir]);

dd= dir([MAT_DIR subdir '*.mat']);
for fn= 1:length(dd),
  fprintf('processing file %s.\n', dd(fn).name);
  fld= who('-file', [MAT_DIR subdir '/' dd(fn).name]);
  load([MAT_DIR subdir '/' dd(fn).name]);
  save([OUT_DIR subdir '/' dd(fn).name], fld{:}, '-V6');
end

end
