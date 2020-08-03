function bbci= bbci_save(bbci, data)
%BBCI_SAVE - Save BBCI classifier
%
%Synopsis:
%  BBCI= bbci_save(BBCI, <DATA>)
%
%To get a description on the structures 'bbci', type
%help bbci_calibrate_structures

% 09-2011 Benjamin Blankertz


bbci= bbci_save_setDefaults(bbci);

if nargin<2,
  data= [];
end
BS= bbci.calibrate.save;

file= fullfile(BS.folder, BS.file);
num= 1;
while ~BS.overwrite && exist(file, 'file'),
  num= num + 1;
  file= fullfile(BS.folder, [BS.file, sprintf('%02d',num)]);
end

save(file, '-STRUCT', 'bbci');
msg= sprintf('Classifier saved to <%s.mat>', file);

if ~isempty(data),
  if ~BS.raw_data,
    data= rmfield(data, {'cnt','mrk','mnt'});
  end
  if isequal(BS.data, 'separately'),
    save([file '_data'], '-STRUCT', 'data');
    msg= [msg ' and data separately to *_data.mat'];
  else
    save(file, 'data', '-APPEND');
  end
end

if BS.figures,
  if ~isfield(data, 'figure_handles'),
    data.figure_handles= sort(findobj('Type','figure'));
  end
  fig_folder= strcat(file, '_figures');
  if ~exist(fig_folder, 'dir'),
    mkdir(fig_folder);
  end
  for ff= data.figure_handles(:)',
    figure(ff);
    fig_name= strrep(get(ff,'Name'), ' ', '_');
    if ispc,
      fig_name= strrep(fig_name, '.', '_');
    end
    filename= fullfile(fig_folder, sprintf('Fig-%02d_%s', ff, fig_name));
    printFigure(filename, BS.figures_spec{:});
  end
  msg= [msg ' and figures in subfolder _figures'];
end
fprintf('%s.\n', msg);
