function printFigure(file, varargin)
% PRINTFIGURE - Save the current Matlab figure
%
%Synopsis:
% printFigure(FILE, <OPT>)
% printFigure(FILE, PAPERSIZE, <OPT>)
%
%Arguments:
% FILE: Name of the output file (without extension).
%   If the filename is relative, EEG_FIG_DIR (global var) is prepended.
% PAPERSIZE: see OPT.paperSize
% OPT: struct or propertyvalue list of optional properties
%  .paperSize: [X Y] size of the output in centimeters, or 'maxAspect'.
%  .device:  Matlab driver used for printing, e.g.
%     ps, eps, epsc, epsc2 (default), jpeg, tiff, png
%  .format: eps or pdf. If OPT.format is 'pdf', then 'epstopdf' is used
%     to convert the output file to PDF format
%     NEW FEATURE: 'format' may be 'svg'. This ignores 'device' and 'paperSize'
%  .append: append current figure as new page in existing file
%  .renderer: how the figure is rendered to a file, 'painters' (default)
%     produces vector images, 'zbuffer' and 'opengl' produce bitmaps

global BBCI_PRINTER EEG_FIG_DIR BCI_DIR

if isnumeric(varargin{1}) || isequal(varargin{1}, 'maxAspect'),
  opt= propertylist2struct(varargin{2:end});
  opt.paperSize= varargin{1};
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, 'paperSize','auto', ...
                       'format', 'eps', ...
                       'device', 'epsc2', ...
                       'folder', EEG_FIG_DIR, ...
                       'prefix', '', ...
                       'resolution', [],...
                       'renderer','painters',...
                       'embed', 1, ...
                       'append',[]);

set(gcf,'Renderer',opt.renderer);
                   
if isfield(opt, 'fig_nos'),
  for ff= 1:length(opt.fig_nos),
    figure(opt.fig_nos(ff));
    if length(opt.fig_nos)>1,
      save_name= [file int2str(ff)];
    else
      save_name= file;
    end
    printFigure(save_name, rmfield(opt, 'fig_nos'));
  end
  return;
end

if ischar(opt.paperSize) && strcmp(opt.paperSize,'maxAspect'),
  set(gcf, 'paperOrientation','landscape', 'paperType','A4', ...
           'paperUnits','inches', ...
           'paperPosition',[0.25 0.97363 10.5 6.5527]);
elseif ischar(opt.paperSize) && strcmp(opt.paperSize,'auto'),
  set(gcf, 'PaperType','A4', ...
           'PaperPositionMode','auto');
else
  if length(opt.paperSize)==2, opt.paperSize= [0 0 opt.paperSize]; end
  set(gcf, 'paperOrientation','portrait', 'paperType','A4', ...
           'paperUnits','centimeters', 'paperPosition',opt.paperSize);
end

if isabsolutepath(file),
  fullName= file;
else
  fullName= [fullfile(opt.folder, opt.prefix) file];
end

if isempty(BBCI_PRINTER) | ~BBCI_PRINTER,
  fprintf('%s not printed (global BBCI_PRINTER not set)\n', fullName);
  if BBCI_PRINTER==1,
    fprintf('press a key to continue');
    pause;
  end
  return;
end

[filepath, filename]= fileparts(fullName); 
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

if strcmpi(opt.format, 'SVG'),
  if ~exist('', 'file'),
    addpath([BCI_DIR 'import/plot2svg']);
  end
  plot2svg([fullName '.svg']);
  return;
end

if isempty(opt.append)
    if isempty(opt.resolution),
      print(['-d' opt.device], fullName);
    else
      print(['-d' opt.device], ['-r' int2str(opt.resolution)], fullName);
    end
else
    % Append figure to existing file
    if isempty(opt.resolution),
      print(['-d' opt.device],'-append', fullName);
    else
      print(['-d' opt.device],'-append', ['-r' int2str(opt.resolution)], fullName);
    end
end

if strcmpi(opt.format, 'PDF') | strcmpi(opt.format, 'EPSPDF'),
  if ~strncmp('eps', opt.device, 3),
    error('For output in PDF format, OPT.device must be eps*');
  end
%  if opt.embed && ~iscluster,
    cmd= sprintf('cd %s; epstopdf --embed %s.eps', filepath, filename);
%  else
%    %% for old epstopdf version on the cluster
%    cmd= sprintf('cd %s; epstopdf %s.eps', filepath, filename);
%  end
  unix_cmd(cmd, 'could not convert EPS to PDF');
  if strcmpi(opt.format, 'PDF'),
    cmd= sprintf('cd %s; rm %s.eps', filepath, filename);
    unix_cmd(cmd, 'could not remove EPS');
  end
end
