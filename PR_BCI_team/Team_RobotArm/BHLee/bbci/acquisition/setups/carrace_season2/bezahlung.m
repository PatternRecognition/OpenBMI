function eur = bezahlung(run)

global VP_CODE;
global TODAY_DIR

D = dir(['D:\cardriver_season2_logfiles\' VP_CODE '*.txt']);

da = inf;
in = 0;
for ii = 1:length(D)
  if ~isequal(D(ii).name(end-4),'t') && D(ii).datenum < da;
    in = ii;
  end
end

eval(['!copy ' 'D:\cardriver_season2_logfiles\' D(in).name ' D:\cardriver_season2_logfiles\tmp.txt'])
fid = fopen(['D:\cardriver_season2_logfiles\tmp.txt']);
for ii = 1:6
  s = fgetl(fid);
end
fclose(fid);

[cnt, mrk] = eegfile_readBV([TODAY_DIR 'carrace_drive' run], 'clab', {'Cz', 'Pz'});
[se in] = ismember(mrk.desc, {'S 64', 'S128'});
in = in(se);

tp = sum(diff(in) == 1);
fp = sum(diff(in) == 0 & in(2:end) == 2);

eur(1) = str2num(s(8:end)) 

eur(2) = 0.05*(tp-fp);

eur = (2/3)*eur;
