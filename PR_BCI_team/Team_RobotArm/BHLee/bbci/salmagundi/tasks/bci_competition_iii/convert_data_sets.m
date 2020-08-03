compet_dir= [DATA_DIR 'eegImport/bci_competition_iii/'];

%% tuebingen
files= {'Competition_train', 'Competition_test'};
for ff= 1:length(files),
  load([compet_dir 'tuebingen/' files{ff}]);
  X= permute(X, [2 1 3]);  %% -> [Ch Tr Time]
  sz= size(X);
  X= reshape(X, [prod(sz(1:2)) sz(3)]);
  save([compet_dir 'tuebingen/' files{ff} '_cnt.txt'], '-ascii', 'X');
  if ff==1,
    save_ascii([compet_dir 'tuebingen/' files{ff} '_lab'], Y, '%d');
  end
  clear X Y
end


%% albany
files= {'A_Train', 'B_Train', 'A_Test', 'B_Test'};
vars= {'Flashing', 'StimulusCode', 'StimulusType', 'TargetChar'};
for ff= 1:length(files),
  file_name= [compet_dir 'albany/Subject_' files{ff}];
  load(file_name);
  Signal= permute(Signal, [3 1 2]);  %% -> [Ch Tr Time]
  sz= size(Signal);
  Signal= reshape(Signal, [prod(sz(1:2)) sz(3)]);
  Signal= double(Signal);
  save([file_name '_Signal.txt'], '-ascii', 'Signal');
  for vv= 1:length(vars),
    if ~exist(vars{vv}, 'var'), continue; end
    eval(sprintf('X= transpose(double(%s));', vars{vv}));
    save_ascii([file_name '_' vars{vv}], X, '%d');
  end
  clear Signal Flashing StimulusCode StimulusType TargetChar
end  


%% martigny
for su= 1:3,
  for ru= 1:4,
    if ru<4,
      run_name= 'train';
    else
      run_name= 'test';
    end
    file_name= sprintf('%s_subject%d_raw%02d', run_name, su, ru);
    X= load([compet_dir 'martigny/' file_name '.asc']);
    if ru<4,
      Y= X(:,end);
      X= X(:,1:end-1);
    end
    nfo.name= file_name;
    nfo.clab= {'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', ...
               'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', ...
               'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', ...
               'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'};
    nfo.fs= 512;
    mnt= projectElectrodePositions(nfo.clab);
    nfo.xpos= mnt.x;
    nfo.ypos= mnt.y;
    save_name= sprintf('%s_subject%d_raw%02d', run_name, su, ru);
    if ru<4,
      save([compet_dir 'martigny/' save_name], 'X', 'Y', 'nfo');
    else
      save([compet_dir 'martigny/' save_name], 'X', 'nfo');
    end
  end
end
for su= 1:3,
  for ru= 1:4,
    if ru<4,
      run_name= 'train';
    else
      run_name= 'test';
    end
    file_name= sprintf('%s_subject%d_psd%02d', run_name, su, ru);
    X= load([compet_dir 'martigny/' file_name '.asc']);
    if ru<4,
      Y= X(:,end);
      X= X(:,1:end-1);
    end
    nfo.name= file_name;
    nfo.clab= {'C3', 'Cz', 'C4', 'CP1', 'CP2', 'P3', 'Pz', 'P4'};
    nfo.fs= 16;
    mnt= projectElectrodePositions(nfo.clab);
    nfo.xpos= mnt.x;
    nfo.ypos= mnt.y;
    save_name= sprintf('%s_subject%d_psd%02d', run_name, su, ru);
    if ru<4,
      save([compet_dir 'martigny/' save_name], 'X', 'Y', 'nfo');
    else
      save([compet_dir 'martigny/' save_name], 'X', 'nfo');
    end
  end
end


%% graz
cd([IMPORT_DIR 'biosig']);
biosig_installer
files= {'k3b', 'k6b', 'l1b', 'O3VR', 'S4b', 'X11b'};
field_list= {'TRIG', 'Classlabel', 'ArtifactSelection'};
for ff= 1:length(files),
  file_name= [compet_dir 'graz/' files{ff}];
  [s,HDR]= sload([file_name '.gdf']);
  save(file_name, 's', 'HDR');
  save([file_name '_s.txt'], '-ascii', 's');
  for fi= 1:length(field_list),
    eval(sprintf('X= double(HDR.%s);', field_list{fi}));
    save_ascii([file_name '_HDR_' field_list{fi}], X, '%d');
  end
end
