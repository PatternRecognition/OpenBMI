global EEG_RAW_DIR
file = [EEG_RAW_DIR 'Timon_05_09_20/imag_lettTimon/'];
file2 = [EEG_RAW_DIR 'Timon_05_09_20/imag_Timon_setup_001_log/'];

d = dir([file '*.log']);

for i = 1:length(d);
  fid = fopen([file d(i).name],'r');
  s = fgets(fid);
  s = s(length('Writing logfile starts at ')+1:end);
  yy = str2num(s(1:4));
  mm = str2num(s(6:7));
  dd = str2num(s(9:10));
  hh = str2num(s(13:14));
  mi = str2num(s(16:17));
  se = str2num(s(19:20));
  ms = str2num(s(22:24));
  s = fgets(fid);
  s = s(length('Used feedback: ')+1:end-1);
  fclose(fid);
  str = sprintf('cp %s%s %s%s_%04d_%02d_%02d_%02d_%02d_%02d_%03d.log',file,d(i).name,file2,s,yy,mm,dd,hh,mi,se,ms);
  system(str);
  str = sprintf('cp %s%s.mat %s%s_%04d_%02d_%02d_%02d_%02d_%02d_%03d.mat',file,d(i).name(1:end-4),file2,s,yy,mm,dd,hh,mi,se,ms);
  system(str);  
end

