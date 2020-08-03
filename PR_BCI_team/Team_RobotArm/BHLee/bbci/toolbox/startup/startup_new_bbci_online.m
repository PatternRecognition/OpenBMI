% We still need some functions of the old online system

global BCI_DIR
save_pathes = {'communication','communication/signalserver'};
for i= 1:length(save_pathes)
  path([BCI_DIR 'online/' save_pathes{i}], path);
end


% ---- -- ---- -- ----


%% In the future, we need only this part

global BCI_DIR DATA_DIR TMP_DIR BBCI_PRINTER
global VP_CODE TODAY_DIR
BBCI_PRINTER= 1;

path(genpath([BCI_DIR 'online_new']), path);

%% Check that some pathes exist
if isempty(DATA_DIR),
  error('DATA_DIR must be defined as global variable');
end

if ~exist(DATA_DIR, 'dir'),
  error('The directory specified as DATA_DIR (-> %s) must exist.', DATA_DIR);
end

if isempty(TMP_DIR),
  TMP_DIR= [DATA_DIR 'tmp/'];
end
if ~exist(TMP_DIR, 'dir'),
  mkdir(TMP_DIR);
end
