function [F,fil] = get_frame(fig,opt);

file = '/tmp/replay_feedback';
if opt.background
  drawnow;
  F.colormap = colormap;
  fil = find_file(file,opt.image_format);
  if ~isempty(opt.other_image_options)
    print(fig, ['-d' opt.image_format], ['-r' int2str(opt.resolution)], ...
          opt.other_image_options{:}, fil);
  else
    print(fig, ['-d' opt.image_format], ['-r' int2str(opt.resolution)], fil);
  end
  pause(1);
  
  F.cdata = imread(fil);
  if isunix
    system(['rm -f ' fil]);
  else
    system(['del ' fil]);
  end
else
  F = getframe(fig);
end
