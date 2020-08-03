function doconcat_movies(F,f,opt);

global BCI_DIR

% find_filename
fil = find_file(f,'mat');
save(fil,'F','f','opt');
fi = separate_dir_file(f);
fi = ['only_for_replay_' fi '_'];
fil2 = find_file([fi],'m');
fil3 = find_file(f,'avi');
fil3b = fil3;
[fil3,di] = separate_dir_file(fil3);
fid = fopen(fil2,'w');
fprintf(fid,'load %s;\n',fil);
fprintf(fid,'cd %s\n',di);
fprintf(fid,'addpath /home/neuro/dornhege/neuro_cvs/matlab/bci/mpgwrite/src/\n');
fprintf(fid,'mpgwrite(F,opt.colormap,''%s'',opt.mpgwrite_options);\n',fil3);
fprintf(fid,'exit\n');
fclose(fid);
fil2b = fil2;
fil2 = separate_dir_file(fil2);
fil2 = fil2(1:end-2);

str = sprintf('!matlab -nojvm -nodisplay -r %s',fil2);
eval(str);

str = sprintf('!rm -f %s %s',fil,fil2b);
eval(str);


% $$$ % mpgwrite ins find_filename
% $$$ mpgwrite(F,opt.colormap,fil,opt.mpgwrite_options);
% $$$ 
% $$$ 
% $$$ % transcode in ein neues find_filename
% $$$ fil2 = find_file(f,'avi');
% $$$ 
% $$$ [fi,di] = separate_dir_file(fil);
% $$$ fi2 = separate_dir_file(fil2);
% $$$ 
% $$$ di2 = pwd;
% $$$ 
% $$$ cd(di);
% $$$ 

fi2 = find_file(f,'avi');
fi2b = fi2;
[fi,di] = separate_dir_file(fil3b);
fi2 = separate_dir_file(fi2);

di2 = pwd;
cd(di);

str = sprintf('%s ',opt.transcode_options{:});

str = sprintf('transcode -i %s -x mpeg2 -V -o %s -y divx5 %s',fi,fi2,str);

system(str);

str = sprintf('rm -f %s\n',fil3b);

system(str);

% merge mit f, falls existiert
if exist([f,'.avi'],'file');
  fil4 = find_file(f,'avi');
  str = sprintf('avimerge -o %s -i %s %s',separate_dir_file(fil4),separate_dir_file([f '.avi']),separate_dir_file(fi2));
  system(str);
  cd(di2);
  
  str = sprintf('rm -f %s',fi2b);
  system(str);
  
  str = sprintf('mv %s %s',fil4,[f,'.avi']);
  system(str);

else
  cd(di2);

  str = sprintf('mv %s %s',fi2b,[f,'.avi']);
  system(str);
end


