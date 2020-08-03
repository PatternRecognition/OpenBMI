function saveFigure_cbw(file, paperSize, bwStyles, fig_nos)
%saveFigure_cbw(file, <paperSize, bwStyles, fig_nos>)
%
% save a color version, and a black-and-white version in an
% (existing) subfolder 'blackNwhite'
%
% SEE  saveFigure

if ~exist('paperSize', 'var'), paperSize=[]; end
if ~exist('bwStyles', 'var'), bwStyles=1; end
if ~exist('fig_nos', 'var'), fig_nos=[gcf]; end

for ff= 1:length(fig_nos),
  figure(fig_nos(ff));
  if length(fig_nos)>1,
    save_name= [file int2str(ff)];
  else
    save_name= file;
  end
  saveFigure(save_name, paperSize);
  [pathstr, name]= fileparts(save_name);
  bw_file= fullfile(pathstr, 'blackNwhite', name);
  saveFigure(bw_file, paperSize, bwStyles);
end
