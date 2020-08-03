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

ff= 2;

filein= [subdir filelist{ff} sbj];
%% 1000Hz Version
[cnt, mrk_orig]= eegfile_readBV([filein '*'], 'clab','Cz');
%mrk_orig= eegfile_readBVmarkers([filein '*']);  %% does not work with '*'

mk= mrkodef_imag_arrow(mrk_orig);
mk= mrk_removeVoidClasses(mk);
mk_rest= mrk_selectClasses(mk.misc, 'cross');
mk_rest.className= {'rest'};
mrk= mrk_mergeMarkers(mk, mk_rest);
mrk= mrk_sortChronologically(mrk);
mrk.pos= [mrk.pos inf];

T= size(cnt.x, 1);
true_y= NaN*zeros(T, 1);

for ii= 1:length(mrk.pos)-1,
  if mrk.pos(ii+1)-mrk.pos(ii)>10000,
    stop_at= mrk.pos(ii) + 1500;
  else
    stop_at= mrk.pos(ii+1)-1;
  end
  true_y(mrk.pos(ii)+999:stop_at)= [-1 1 0]*mrk.y(:,ii);
end


fileout= [folder savelist{ff} appendix(vp) '_1000Hz'];
save_ascii([DATA_DIR 'bci_competition_iv/' ...
            savelist{ff} appendix(vp) '_1000Hz_true_y'], ...
           true_y);
save([DATA_DIR 'bci_competition_iv/' ...
      savelist{ff} appendix(vp) '_1000Hz_true_y'], ...
     'true_y', '-V6');
end
