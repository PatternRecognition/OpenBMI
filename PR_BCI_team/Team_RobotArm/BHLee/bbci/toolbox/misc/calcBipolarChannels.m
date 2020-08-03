function dat= calcBipolarChannels(dat)
%dat= calcBipolarChannels(dat)
%
% replacements:
%  F9 for EOGhp, F10 for EOGhn, Fp2 for EOGvn

delChans= chanind(dat, 'EMGfn', 'EOGvp');
EMGlp= chanind(dat, 'EMGlp');
if isempty(EMGlp), return; end
EMGln= chanind(dat, 'EMGln');
EMGrp= chanind(dat, 'EMGrp');
EMGrn= chanind(dat, 'EMGrn');
EMGfp= chanind(dat, 'EMGfp');
EMGfn= chanind(dat, 'EMGfn');
EOGhp= chanind(dat, 'EOGhp');
if isempty(EOGhp), 
  EOGhp= chanind(dat, 'F9'); 
else
  delChans= [delChans, EOGhp];
end
EOGhn= chanind(dat, 'EOGhn');
if isempty(EOGhn), 
  EOGhn= chanind(dat, 'F10'); 
else
  delChans= [delChans, EOGhn];
end
EOGvp= chanind(dat, 'EOGvp');
EOGvn= chanind(dat, 'EOGvn');
if isempty(EOGvp),
  EOGvp= chanind(dat, 'Fp2'); 
  delChans= [delChans, EOGvn];
elseif isempty(EOGvn),
  EOGvn= chanind(dat, 'Fp2'); 
else
  delChans= [delChans, EOGvn];
end
RESp= chanind(dat, 'RESp');

dat.clab{EMGlp}= 'EMGl';
dat.x(:,EMGlp,:)= dat.x(:,EMGlp,:) - dat.x(:,EMGln,:);
dat.clab{EMGrp}= 'EMGr';
dat.x(:,EMGrp,:)= dat.x(:,EMGrp,:) - dat.x(:,EMGrn,:);
if ~isempty(EMGfp), 
  dat.clab{EMGfp}= 'EMGf';
  dat.x(:,EMGfp,:)= dat.x(:,EMGfp,:) - dat.x(:,EMGfn,:);
end
dat.clab{EMGln}= 'EOGh';
dat.x(:,EMGln,:)= dat.x(:,EOGhp,:) - dat.x(:,EOGhn,:);
dat.clab{EMGrn}= 'EOGv';
dat.x(:,EMGrn,:)= dat.x(:,EOGvp,:) - dat.x(:,EOGvn,:);
if ~isempty(RESp),
  RESn= chanind(dat, 'RESn');
  dat.x(:,RESp,:)= dat.x(:,RESp,:) - dat.x(:,RESn,:);
  dat.clab{RESp}= 'RES';
  delChans=  [delChans, RESn];
end


remChans= setdiff(1:length(dat.clab), delChans);

dat.clab= {dat.clab{remChans}};
dat.x= dat.x(:,remChans,:);
