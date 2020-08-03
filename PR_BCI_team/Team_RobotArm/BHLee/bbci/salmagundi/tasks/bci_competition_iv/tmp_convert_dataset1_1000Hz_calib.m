folder= [DATA_DIR 'eegExport/bci_competition_iv/'];

dd= [BCI_DIR 'salmagundi/tasks/bci_competition_iv/'];
subdir_list= textread([dd 'session_list'], '%s');

filelist= {'imag_arrow', 'imag_audicompetition'};
savelist= {'BCICIV_calib_ds1', 'BCICIV_eval_ds1'};
appendix= 'fbagced';

vers = version;
if (str2double(vers(1)) == 7)
  opt_save= {'-V6'};
else
  opt_save= {};
end

for vp= 1:length(subdir_list),

subdir= [subdir_list{vp} '/'];
is= min(find(subdir=='_'));
if strcmp(subdir(1:4),'BCIC'),
  is= length(subdir);
end
sbj= subdir(1:is-1);
fprintf('** processing %s.\n', subdir);

for ff= 1,

filein= [subdir filelist{ff} sbj];

%% 1000Hz Version
[ct, mrk_orig]= eegfile_readBV([filein '*'], ...
                               'clab',{'not','E*','Fp2','F9','F10'});
cnt= int16(ct.x*10);
mk= mrkodef_imag_arrow(mrk_orig);
mk= mrk_removeVoidClasses(mk);
mrk= struct('pos', mk.pos);
mrk.y= [-1 1]*mk.y;
mnt= setElectrodeMontage(ct.clab);
nfo= strukt('fs',ct.fs, ...
            'classes', mk.className, ...
            'clab',ct.clab, ...
            'xpos', mnt.x, ...
            'ypos', mnt.y);
clear ct

fileout= [folder savelist{ff} appendix(vp) '_1000Hz'];
if ff==1,
  save(fileout, 'cnt','mrk','nfo',opt_save{:});
  save_ascii([fileout '_mrk'], [mrk.pos; mrk.y]');
else
  save(fileout, 'cnt','nfo',opt_save{:});
  save_ascii([DATA_DIR 'bci_competition_iv/' ...
              savelist{ff} appendix(vp) '_1000Hz_mrk'], ...
             [mrk.pos; mrk.y]');
end
save_ascii([fileout '_cnt'], cnt);
fid= fopen([fileout '_nfo.txt'], 'w');
fprintf(fid, 'fs: %d\n', nfo.fs);
fprintf(fid, 'classes: %s\n', vec2str(nfo.classes));
fprintf(fid, 'clab: %s\n', vec2str(nfo.clab));
fprintf(fid, 'xpos: %s\n', vec2str(nfo.xpos, '%g'));
fprintf(fid, 'ypos: %s\n', vec2str(nfo.xpos, '%g'));
fclose(fid);
clear cnt

end

end
