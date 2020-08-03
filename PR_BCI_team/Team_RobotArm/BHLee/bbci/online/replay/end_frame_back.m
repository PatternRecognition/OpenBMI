function end_frame(movie,opt);

if isunix
  movie.fields = movie.fields(1:movie.pointer);
  
  if ~isempty(movie.fields)
    if isempty(opt.colormap)
      opt.colormap = movie.fields(end).colormap;
    end
    doconcat_movies(movie.fields,movie.file,opt);
  end
else
  try
    movie = close(movie);
  end
end
