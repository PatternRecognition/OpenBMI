function mrk= mrkdef_imag_fbbasket(Mrk, file, opt)

%Markers:
% 1-9: target no
% 10:  hit
% 11:  miss
% 209: play
% 210: countdown finished
% 211: pause

%% Mrko is only used to achieve high accuracy for field 'duration'
Mrko= readMarkerTable(file, 'raw');
Mrk.toe= Mrko.toe;
Mrk.pos= ceil(Mrko.pos/Mrko.fs*Mrk.fs);

nTargets= max(intersect(unique(Mrk.toe), 1:9));
className= cellstr([repmat('target ', [nTargets 1]), int2str((1:nTargets)')])';


classDef= cat(1, num2cell([1:nTargets 10 11]), ...
              {className{:}, 'hit', 'miss'});
mrk= makeClassMarkers(Mrk, classDef,0,0);
Mrko= makeClassMarkers(Mrko, classDef,0,0);

%% check consistency - TODO: this is only a poor trying to fix inconsistencies
it= any(mrk.y(1:nTargets,:),1);
start= 1;
while start<length(it)-10 & (~all(it(start:2:end)) | any(it(start+1:2:end))),
  start= start + 1;
end

if start>1,
  fprintf('%d markers skipped due to stimulus/response inconsistency\n', ...
          start-1);
  mrk= mrk_selectEvents(mrk, start:length(it));
  Mrko= mrk_selectEvents(Mrko, start:length(it));
end

iStim= find(any(mrk.y(1:nTargets,:),1));
iResp= find(any(mrk.y(nTargets+1:end,:),1));
mrk.ishit= [mrk.y(nTargets+1,2:end), 0];
mrk= mrk_addIndexedField(mrk, 'ishit');
mrk= mrk_selectClasses(mrk, 'target*');
mrk= mrk_addRunNumbers(mrk, file);
mrk.duration= (Mrko.pos(iResp)-Mrko.pos(iStim))/Mrko.fs*1000;
mrk= mrk_addIndexedField(mrk, 'run_no');
