function writeCsvOriginalSettings(outfile, infile)
% writeCsvOriginalSettings - write the original settings in CSV format
%
% usage:
%  writeCsvOriginalSettings(outfile, infile)
% or
%  writeCsvOriginalSettings(1, infile)   [dump on the screen]
%
% example infile
%  infile = [BCI_DIR 'studies/season2/session_list_ext']
%  infile = [BCI_DIR 'studies/season3/session_list']


subdir_list= textread(infile, '%s');


if isequal(outfile,1),
  fid= 1;
else
  fid = fopen(outfile, 'w');
end

fprintf(fid, 'subject_date,classes,band,ival,threshold,used patterns,clab,message\n');
for vp= 1:length(subdir_list),
  sub_dir= [subdir_list{vp} '/'];
  is= min(find(sub_dir=='_'));
  sbj= sub_dir(1:is-1);
 
  try
    %% load original settings that have been used for feedback
    bbci= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','bbci');
  catch
    fprintf(fid, '%s,,,,,,\n',sbj);
    continue;
  end

  f=inline('[x '' '']');

  %% get the two classes that have been used for feedback
  classes= bbci.classes;

  %% get the time interval and channel selection that has been used
  band = bbci.setup_opts.band;
  ival = bbci.setup_opts.ival;
  thre = bbci.setup_opts.threshold;
  patt = bbci.setup_opts.usedPat;
  nPat = bbci.setup_opts.nPat;

  clab = bbci.setup_opts.clab;

  msg  = strrep(bbci.analyze.message, char(10), ' ');
  fprintf(fid, '%s,%s,[%g %g],[%g %g],%g,%s/%d,"%s",%s\n',...
          subdir_list{vp}, vec2str(classes, '%s', ' '), band(1), band(2), ...
          ival(1), ival(2), thre, num2str(patt), 2*nPat, vec2str(clab), msg);

end

if fid~=1,
  fclose(fid);
end
