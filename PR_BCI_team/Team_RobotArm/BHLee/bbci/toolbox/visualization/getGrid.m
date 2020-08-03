function grid= getGrid(displayMontage)
%grid= getGrid(displayMontage)
%
% IN   displayMontage - any *.mnt file in EEG_CFG_DIR, 
%                       e.g., 'small', 'medium', 'large',
%                       or a string defining the montage
%
% OUT  grid           - 2-d cell array containing the channel labels
%
% GLOBZ EEG_CFG_DIR

global EEG_CFG_DIR;

if ismember(',', displayMontage) | ismember(sprintf('\n'), displayMontage),
  readFcn= 'strread';
  montage= displayMontage;
else
  readFcn= 'textread';
  montage= fullfile(EEG_CFG_DIR, [displayMontage '.mnt']);
  if ~exist(montage, 'file'),
    error(sprintf('unknown montage (checked %s)', ...
                  montage));
  end
end
grid= feval(readFcn, montage, '%s');
width= 1 + sum(grid{1}==',');
grid= cell(1, width);
[grid{:}]= feval(readFcn, montage, repmat('%s',1,width), 'delimiter',',');
grid= [grid{:}];
