function startup_public_bbci(keep_old_toolbox)

global BCI_DIR BBCI_DIR

if isempty(BBCI_DIR),
  %% try to guess to location
  REPOS_DIR= fileparts(BCI_DIR);
  guess_list= {'~/git/public', 
               '~/git/bbci_public',
               '~/svn/public',
               '~/svn/bbci_public',
               fullfile(REPOS_DIR, 'public'),
               fullfile(REPOS_DIR, 'bbci_public')};
  for k= 1:length(guess_list),
    BBCI_DIR= guess_list{k};
    if exist(BBCI_DIR, 'dir'),
      break;
    end
  end
  if ~exist(BBCI_DIR, 'dir'),
    error('Could not guess location - specify BBCI_DIR as global variable');
  end
end

if nargin==0 || ~keep_old_toolbox,
  % remove old toolbox
  wstat= warning('off');
  rmpath(genpath(fullfile(BCI_DIR, 'toolbox')));
  rmpath(genpath(fullfile(BCI_DIR, 'acquisition')));
  rmpath(genpath(fullfile(BCI_DIR, 'online')));
  rmpath(genpath(fullfile(BCI_DIR, 'online_new')));
  warning(wstat);
  path(path, fullfile(BCI_DIR, 'toolbox', 'startup'));
end

run(fullfile(BBCI_DIR, 'startup_bbci_toolbox'));
