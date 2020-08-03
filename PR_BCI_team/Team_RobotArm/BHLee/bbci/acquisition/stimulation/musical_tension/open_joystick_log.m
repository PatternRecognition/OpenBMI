function log_fid=open_joystick_log(audiofile)

global TODAY_DIR BCI_DIR

%TODAY_DIR=[BCI_DIR 'acquisition' filesep 'stimulation' filesep 'musical_tension' filesep 'log_data' filesep];
%logfile anlegen
filename=[TODAY_DIR audiofile];

header=['# Joystick data of tension ratings - ' datestr(now,'yyyy-mm-dd HH:MM:SS.FFF') '\n # Audiofile: ' audiofile '\n'];
logfile= [filename '001.txt'];
num= 1;
while exist(logfile, 'file'),
      num= num + 1;
      logfile= sprintf('%s%03d.txt', filename, num);
end
 
  log_fid= fopen(logfile, 'w');
  %fwrite(log_fid,header);
  fprintf(log_fid,  header);
  if log_fid==-1,
    error(sprintf('could not open log-file <%s> for writing', logfile));
  end