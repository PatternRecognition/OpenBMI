force= 0;

dd= [BCI_DIR 'salmagundi/tasks/bci_competition_iv/'];
subdir_list= textread([dd 'session_list'], '%s');

grd= sprintf(['EOGh,scale,F3,Fz,F4,legend,EOGv\n' ...
              'C5,C3,C1,Cz,C2,C4,C6\n' ...
              'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
              'EMGl,P5,P3,EMGf,P4,P6,EMGr']);

filelist= {'imag_arrow', 'imag_audicompetition'}; 

for vp= 1:length(subdir_list),

subdir= [subdir_list{vp} '/'];
is= min(find(subdir=='_'));
if strcmp(subdir(1:4),'BCIC'),
  is= length(subdir);
end
sbj= subdir(1:is-1);

if exist([EEG_MAT_DIR subdir], 'dir') & ~force,
  fprintf('** %s is already processed.\n', subdir);
  continue;
end

fprintf('** processing %s.\n', subdir);

for ff= 1:length(filelist),

file= [subdir filelist{ff} sbj];
hdr= eegfile_readBVheader(file);
Wps= [42 49]/hdr.fs*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 40);
[filt.b, filt.a]= cheby2(n, 50, Ws);
if ~strcmp(subdir(1:4),'BCIC'),
  bip_list= {{'EOGvp','Fp2','EOGv'}, {'F9','F10','EOGh'}};
  ld= procutil_biplist2projection(hdr, bip_list, 'delete_policy','second');
  [cnt, mrk_orig]= eegfile_loadBV([file '*'], 'fs',100, 'filt',filt, ...
                                  'linear_derivation',ld);
else
  [cnt, mrk_orig]= eegfile_loadBV([file '*'], 'fs',100, 'filt',filt);
end

cnt.title= file;
%mrk= feval(['mrkodef_' filelist{ff}], mrk_orig);
mrk= mrkodef_imag_arrow(mrk_orig);
mnt= setElectrodeMontage(cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

fs_orig= mrk_orig.fs;
var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig};
eegfile_saveMatlab(file, cnt, mrk, mnt, ...
                   'channelwise',1, ...
                   'format','int16', ...
                   'resolution', NaN, ...
                   'vars', var_list);

end

end
