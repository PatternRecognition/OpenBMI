function [movie, number] = add_frame(movie, F, opt, file, number)

if isunix,
  sz= movie.Height*movie.Width*movie.TotalFrames*3;
else
  if number==1,
    filename= [file '.avi'];
  else
    filename= sprintf('%s_%03d.avi', file, number);
  end
  sz = dir(filename);
  sz = sz.bytes;
end
if sz>=opt.maxFileSize,
  end_frame(movie, opt);
  if number==1,
    if isunix,
      stat= unix(sprintf('mv %s.avi %s_001.avi', file, file));
      if stat~=0,
        keyboard;
      end
    else
      cmd= sprintf('rename %s.avi %s_001.avi', file, file);
      dos(cmd);
    end
  end    
  [pathstr, filestr]= fileparts(file);
  cmd= sprintf('cd %s; transcode -i %s_%03d.avi -use_rgb -z -y xvid -o tmp_%s_%03d.avi', pathstr, filestr,number,filestr,number);
  unix(cmd);
  stat= unix(sprintf('rm -f %s_%03d.avi', file,number));

  number = number+1;
  movie = avifile(sprintf('%s_%03d.avi',file,number), 'fps',opt.fps, ...
                  'Compression',opt.compression,'quality',opt.quality);
end

movie = addframe(movie,F);
