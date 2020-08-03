subdir_list= {'VPiz_08_06_12/',  'VPip_08_06_19/'};
file_list= {'sssep_BCI_27_new_lev6VPiz', 'sssep_BCI_27_new_lev8_difVPip'};

for vp= 1:length(subdir_list),

  file= [subdir_list{vp} file_list{vp}];
  hdr= eegfile_readBVheader(file);
  Wps= [40 49]/hdr.fs*2;
  [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
  [filt.b, filt.a]= cheby2(n, 50, Ws);
  bip_list= {{'EOGvp','Fp2','EOGv'}, {'F9','F10','EOGh'}};
  ld= procutil_biplist2projection(hdr, bip_list, 'delete_policy','second');
  
  clear cnt
  [cnt,mrk_orig]= eegfile_loadBV([file '*'], 'fs',100, 'filt',filt, ...
                                 'linear_derivation',ld);
  stimDef = {'S  1','S  2';'Left','Right'};
  mrk= mrk_defineClasses(mrk_orig, stimDef);
  fs_orig= mrk_orig.fs;
  mnt= getElectrodePositions(cnt.clab);
  mnt= mnt_setGrid(mnt, 'medium');
  
  
  eegfile_saveMatlab(file, cnt, mrk, mnt, ...
                     'channelwise',1, ...
                     'format','int16', ...
                     'resolution', NaN, ...
                     'vars',{'fs_orig',fs_orig, 'mrk_orig',mrk_orig});
end
