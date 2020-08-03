function saveFigure(file, paperSize, bwStyles, device, fig_nos)
%saveFigure(file, <paperSize, bwStyles, device='epsc2'>)
%
% IN   file      - file name (no extension),
%                  relative to EEG_FIG_DIR unless beginning with 'filesep'
%      paperSize - in centimeters or 'maxAspect' (default)
%      bwStyles  - see col2bw
%
% GLOBZ EEG_FIG_DIR

if ~exist('paperSize','var') | isempty(paperSize), paperSize='maxAspect'; end
if ~exist('device','var') | isempty(device), device='epsc2'; end
if exist('fig_nos','var'),
  for ff= 1:length(fig_nos),
    figure(fig_nos(ff));
    if length(fig_nos)>1,
      save_name= [file int2str(ff)];
    else
      save_name= file;
    end
    saveFigure(save_name, paperSize, bwStyles, device);
  end
  return;
end

if ischar(paperSize) & strcmp(paperSize,'maxAspect'),
  set(gcf, 'paperOrientation','landscape', 'paperType','A4', ...
           'paperUnits','inches', ...
           'paperPosition',[0.25 0.97363 10.5 6.5527]);
else
  if length(paperSize)==2, paperSize= [0 0 paperSize]; end
  set(gcf, 'paperOrientation','portrait', 'paperType','A4', ...
           'paperUnits','centimeters', 'paperPosition',paperSize);
end

if file(1)==filesep,
  fullName= file;
elseif filesep=='\' & file(2)==':'
  fullName = file;
else
  global EEG_FIG_DIR
  fullName= [EEG_FIG_DIR file];
end

filepath= fileparts(fullName);        %% create directory if necessary
if ~exist(filepath, 'dir'),
  [parentdir, newdir]= fileparts(filepath);
  [status, msg]= mkdir(parentdir, newdir);
  if status~=1,
    error(msg);
  end
  if isunix,
    unix(sprintf('chmod a-rwx,ug+rwx %s', filepath));
  end
  fprintf('new directory <%s%s%s> created\n', parentdir, filesep, newdir);
end

if exist('bwStyles', 'var') & ~isempty(bwStyles),
  [cols,hl]= col2bw(bwStyles);
  cmap= colormap;
  cmap_bw= cmap(:,1)*[1 1 1];
  %% change colormap only if non-gray colors are in use
  if any(cmap(:)~=cmap_bw(:)),
    colormap('gray');
  end
  print('-deps2', fullName);
  colormap(cmap);
  bw2col(cols,hl);
else
  print(['-d' device], fullName);
end
