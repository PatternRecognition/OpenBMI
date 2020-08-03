function movie = add_frame(movie,F,opt);

if isunix
  movie.pointer = movie.pointer+1;
  movie.fields(movie.pointer).cdata = F.cdata;
  movie.fields(movie.pointer).colormap = F.colormap;
  if movie.pointer == opt.max_size;
    movie.pointer = 0;
    if isempty(opt.colormap)
      opt.colormap = movie.fields(end).colormap;
    end
    doconcat_movies(movie.fields,movie.file,opt);
    movie.fields = struct('cdata',cell(1,opt.max_size),'colormap',cell(1,opt.max_size));;
 end
% $$$   movie.pointer = movie.pointer+1;
% $$$   movie.movie = addframe(movie.movie,F);
% $$$   if movie.pointer == opt.max_size;
% $$$     keyboard
% $$$     movie.movie = close(movie.movie);
% $$$     doconcat_movies3(movie.movie,movie.file,movie.filewrite,opt);
% $$$     fi = find_file(movie.file,'.avi');
% $$$     movie.movie = avifile(fi,'fps',opt.fps,'Compression','none');
% $$$     movie.pointer = 0;
% $$$     movie.filewrite = fi;
% $$$   end
  
else
  movie = addframe(movie,F);
end




