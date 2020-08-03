export_dir= '/home/neuro/data/dropbox/blanker/competition_iii/';
[dummy, subbase]= readDatabase;
opt_load= struct('clab', {{'not', 'E*'}}, ...
                 'types', {{'Stimulus'}});
cd([BCI_DIR 'tasks/bci_competition_iii/'])

dir_list= {'Falk_04_03_31', ...
           'Gabriel_04_03_30', ...
           'Guido_04_03_29', ...
           'Klaus_04_04_08', ...
           'Matthias_04_03_24'};

classDef= {1, 2, 3; 'left', 'right', 'foot'};

fs= 100;

for dd= 1:length(dir_list),
  dir_name= dir_list{dd};
  is= find(dir_name=='_');
  subject= dir_name(1:is-1);
  file= strcat(dir_name, '/', {'imag_move', 'imag_lett'}, subject);
  code= getSubjectIndex(subbase, subject, 'code')
  [cnt, Mrk]= loadGenericEEG_int(file, opt_load, 'fs',fs);
  clear cnt
  mrk= makeClassMarkers(Mrk, classDef, 0, 0);
  mrk= struct('pos', mrk.pos, ...
              'y', [1 2 3]*mrk.y, ...
              'className', {mrk.className});
    
  save([export_dir 'mrk_all_' code], 'mrk');
end
