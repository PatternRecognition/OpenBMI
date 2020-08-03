if ~exist('test','var')
  %test = 'classification';
  test =  'transfer';
end

%% Load true labels
load(['true_y_' test]);

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

dir = [DATA_DIR 'results/alternative_' test '/'];

wt=what(dir);

names = foreach(inline('x(1:end-4)','x'),wt.mat);

table = nan*ones(length(subdir_list),length(names));
btable = nan*ones(length(subdir_list),length(names));
for i=1:length(names)
  file = [dir names{i}];
  try
    S=load(file);
  catch
    fprintf('[%d] %s not found.\n', i, file);
  end
  table(:,i)=S.perf';
  
  for j=1:length(subdir_list)
    lab = [true_y{j}<0; true_y{j}>0];
    if isfield(S,'memo') & ~isempty(S.memo{j}) & isfield(S.memo{j}, 'out')
      btable(j,i)=bitrate(mean(loss_0_1(lab, S.memo{j}.out)));
    end
  end
    
end
table=table*100;


