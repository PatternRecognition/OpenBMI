function saveFigure_png(file, paperSize, fig_nos)
%saveFigure_png(file, paperSize, <fig_nos>)
%
% saves (additionally to the EPS version) the current figure 
% in PNG format in a subfolder 'png'


if ~exist('paperSize', 'var'), paperSize=[]; end
if ~exist('fig_nos', 'var'), fig_nos=gcf; end

save_name= file;
for ff= 1:length(fig_nos),
  figure(fig_nos(ff));
  if length(fig_nos)>1,
    save_name= [file int2str(ff)];
  end
  saveFigure(save_name, paperSize);
  [pathstr, name]= fileparts(save_name);
  png_file= fullfile(pathstr, 'png', name);
  saveFigure(png_file, paperSize, [], 'png');
end
