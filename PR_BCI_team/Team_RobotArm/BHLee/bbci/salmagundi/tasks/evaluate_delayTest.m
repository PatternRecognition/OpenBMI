function evaluate_delayTest(filespec)

dd= dir(strcat(filespec, '.vmrk'));
pathstr= fileparts(filespec);
filelist= {dd.name};

for ii= 1:length(filelist),
  suplot(length(filelist), ii);
  file= filelist{ii}(1:end-5);
  mrk= eegfile_readBVmarkers([pathstr '/' file], 0);
  iS= strmatch('S  2', mrk.desc);
  iR= strmatch('S  1', mrk.desc);
%  [length(iS), length(iR)]
  latency= (mrk.pos(iR) - mrk.pos(iS)) / mrk.fs*1000;
  if any(latency)<0,
    error('mrk mismatch');
  end
  fprintf('%20s:  median: %6.1f,  mean: %6.1f +/- %5.1f\n', ...
          file, median(latency), mean(latency), std(latency));
  hist(latency, 30);
  title(file);
end
