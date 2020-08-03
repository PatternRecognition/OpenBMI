export_dir= [DATA_DIR 'eegImport/bci_competition_iii/berlin/'];
[dummy, subbase]= readDatabase;
opt_load= struct('clab', {{'not', 'E*'}}, ...
                 'types', {{'Stimulus'}});
cd([BCI_DIR 'tasks/bci_competition_iii/'])

dir_list= {'Falk_04_03_31', ...
           'Gabriel_04_03_30', ...
           'Guido_04_03_29', ...
           'Klaus_04_04_08', ...
           'Matthias_04_03_24'};
train_perc= [30 60 80 10 20];

classDef= {2, 3; 'right', 'foot'};

data_name= 'data_set_IVa';

for fs= [100, 1000],
export_dir_fs= sprintf('%s%dHz/', export_dir, fs);

for dd= 1:length(dir_list),
  dir_name= dir_list{dd};
  is= find(dir_name=='_');
  subject= dir_name(1:is-1);
  file= strcat(dir_name, '/', {'imag_move', 'imag_lett'}, subject);
  [cnt, Mrk]= loadGenericEEG_int(file, opt_load, 'fs',fs);
  mnt= projectElectrodePositions(cnt.clab);
  code= getSubjectIndex(subbase, subject, 'code');
  nfo= struct('name', [data_name '_' code], ...
              'fs', cnt.fs, ...
              'clab', {cnt.clab}, ...
              'xpos', mnt.x, ...
              'ypos', mnt.y);
  mrk= makeClassMarkers(Mrk, classDef, 0, 0);
  mrk= struct('pos', mrk.pos, ...
              'y', [1 2]*mrk.y, ...
              'className', {mrk.className});
  
  nEpochs= length(mrk.y);
  tp= train_perc(dd)/100*nEpochs;
  test_idx= tp+1:nEpochs;
  fprintf('test - 1:%d 2:%d\n', ...
          sum(mrk.y(test_idx)==1), sum(mrk.y(test_idx)==2));

  true_y= mrk.y;
  save([export_dir_fs data_name '_' code '_truth'], 'true_y');
  mrk.y(test_idx)= NaN;
  
  cnt= cnt.x;
  save([export_dir_fs nfo.name], 'cnt', 'mrk', 'nfo');
  save_ascii_scaled([export_dir_fs nfo.name '_cnt.txt'], cnt, '%.1f', 0.1);
  save_ascii([export_dir_fs nfo.name '_mrk.txt'], [mrk.pos; mrk.y]', '%d');
  fid= fopen([export_dir_fs nfo.name '_nfo.txt'], 'w');
  fprintf(fid, 'name: %s\n', nfo.name);
  fprintf(fid, 'fs: %d\n', nfo.fs);
  fprintf(fid, 'clab: %s\n', vec2str(nfo.clab));
  fprintf(fid, 'xpos: %s\n', vec2str(nfo.xpos, '%g'));
  fprintf(fid, 'ypos: %s\n', vec2str(nfo.xpos, '%g'));
  fclose(fid);
  clear cnt
end


data_name= 'data_set_IVb';
train_file= 'Guido_04_11_01/imag_lettGuido';
trainDef= {1, 2; 'left', 'foot'};
test_file= 'Guido_04_11_01/imag_auditoryGuido';
testDef= {1, 2, 3, 100; 'left', 'foot', 'relax', 'stop'};
subject= 'Guido';
code= getSubjectIndex(subbase, subject, 'code');

%% training data
[cnt, Mrk]= loadGenericEEG_int(train_file, opt_load, 'fs',fs);
mrk= makeClassMarkers(Mrk, trainDef, 0, 0);
mnt= projectElectrodePositions(cnt.clab);
nfo= struct('name', [data_name '_' code '_train'], ...
            'fs', cnt.fs, ...
            'clab', {cnt.clab}, ...
            'xpos', mnt.x, ...
            'ypos', mnt.y);

mrk= struct('pos', mrk.pos, ...
            'y', [-1 1]*mrk.y);
cnt= cnt.x;
save([export_dir_fs nfo.name], 'cnt', 'mrk', 'nfo');
save_ascii_scaled([export_dir_fs nfo.name '_cnt.txt'], cnt, '%.1f', 0.1);
save_ascii([export_dir_fs nfo.name '_mrk.txt'], [mrk.pos; mrk.y]', '%d');
fid= fopen([export_dir_fs nfo.name '_nfo.txt'], 'w');
fprintf(fid, 'name: %s\n', nfo.name);
fprintf(fid, 'fs: %d\n', nfo.fs);
fprintf(fid, 'clab: %s\n', vec2str(nfo.clab));
fprintf(fid, 'xpos: %s\n', vec2str(nfo.xpos, '%g'));
fprintf(fid, 'ypos: %s\n', vec2str(nfo.xpos, '%g'));
fclose(fid);
clear cnt

%% test data
nfo.name= [data_name '_' code '_test'];
mtab= readMarkerTable(test_file, fs);
is= find(mtab.toe==252);
ie= find(mtab.toe==253);
from= mtab.pos(is)*1000/fs - fs;
len= mtab.pos(ie)*1000/fs + fs - from;
[cnt, Mrk]= loadGenericEEG_int(test_file, opt_load, 'fs',fs, ...
                               'from',from, 'maxlen',len);
Mrk= makeClassMarkers(Mrk, testDef, 0, 0);
mrk= mrk_selectClasses(Mrk, testDef(2, 1:3));
idx= find(Mrk.y(4,:));
mrk.length= Mrk.pos(idx) - Mrk.pos(idx-1);

delay_ms= 500;
delay= delay_ms/1000*mrk.fs;
true_y= NaN*zeros(size(cnt.x,1), 1);
template= zeros(size(cnt.x,1), 1);
for ii= 1:length(mrk.pos),
  iv= mrk.pos(ii) + [0:mrk.length(ii)] + delay;
  true_y(iv)= [-1 1 0]*mrk.y(:,ii);  
  template(iv)= 1;
end
save([export_dir_fs nfo.name '_truth'], 'true_y', 'template');

cnt= cnt.x;
save([export_dir_fs nfo.name], 'cnt', 'nfo');
save_ascii_scaled([export_dir_fs nfo.name '_cnt.txt'], cnt, '%.1f', 0.1);
fid= fopen([export_dir_fs nfo.name '_nfo.txt'], 'w');
fprintf(fid, 'name: %s\n', nfo.name);
fprintf(fid, 'fs: %d\n', nfo.fs);
fprintf(fid, 'clab: %s\n', vec2str(nfo.clab));
fprintf(fid, 'xpos: %s\n', vec2str(nfo.xpos, '%g'));
fprintf(fid, 'ypos: %s\n', vec2str(nfo.xpos, '%g'));
fclose(fid);
clear cnt




data_name= 'data_set_IVc';
test_file= 'Guido_04_11_01/imag_lettfastGuido';
classDef= {1, 2, 3; 'left', 'foot','relax'};
subject= 'Guido';
code= getSubjectIndex(subbase, subject, 'code');

[cnt, Mrk]= loadGenericEEG_int(test_file, opt_load, 'fs',fs);
mrk= makeClassMarkers(Mrk, classDef, 0, 0);
mnt= projectElectrodePositions(cnt.clab);
nfo= struct('name', [data_name '_' code '_test'], ...
            'fs', cnt.fs, ...
            'clab', {cnt.clab}, ...
            'xpos', mnt.x, ...
            'ypos', mnt.y);

true_y= [-1 1 0]*mrk.y;
save([export_dir_fs nfo.name '_truth'], 'true_y');

mrk= struct('pos', mrk.pos);
cnt= cnt.x;
save([export_dir_fs nfo.name], 'cnt', 'mrk', 'nfo');
save_ascii_scaled([export_dir_fs nfo.name '_cnt.txt'], cnt, '%.1f', 0.1);
save_ascii([export_dir_fs nfo.name '_mrk.txt'], mrk.pos', '%d');
fid= fopen([export_dir_fs nfo.name '_nfo.txt'], 'w');
fprintf(fid, 'name: %s\n', nfo.name);
fprintf(fid, 'fs: %d\n', nfo.fs);
fprintf(fid, 'clab: %s\n', vec2str(nfo.clab));
fprintf(fid, 'xpos: %s\n', vec2str(nfo.xpos, '%g'));
fprintf(fid, 'ypos: %s\n', vec2str(nfo.xpos, '%g'));
fclose(fid);
clear cnt

end %% for fs






return






epo= makeEpochs(cnt, mrk, [0 1500]);
epo= proc_baseline(epo, [0 150]);
grid_plot(epo, mnt);



%% demo

file= '/mnt/share/Daten/bci_competition_iii/berlin/data_set_IVa_av.nfo';
eval(char(textread(file, '%s', 'whitespace','', 'bufsize',10000)));

fprintf('%s\n', nfo.name);
fprintf('labelled: <%s/%s>: %d/%d epochs\n', ...
        mrk.className{:}, sum(mrk.y==1), sum(mrk.y==2));
fprintf('unlabelled: %d epochs\n', sum(isnan(mrk.y)));

plot(nfo.xpos, nfo.ypos, 'o', 'markerFaceColor',[1 0.6 0.6], 'markerEdgeColor','none', 'markerSize',18); 
text(nfo.xpos, nfo.ypos, nfo.clab, 'horizontalAli','center'); 
axis([-1 1 -1 1], 'square', 'off');
