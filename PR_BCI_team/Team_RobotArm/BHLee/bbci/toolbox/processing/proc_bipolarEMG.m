function dat= proc_bipolarEMG(dat)
%dat= proc_bipolarEMG(dat)
%
% special use only

% for two player mode
if ~isempty(strpatternmatch('x*', dat.clab)),
%  dat1= proc_selectChannels(dat, 'not','x*');
%  dat1= proc_bipolarEMG(dat);
  dat2= proc_selectChannels(dat, 'x*');
  for k= 1:length(dat2.clab),
    dat2.clab{k}= dat2.clab{k}(2:end);
  end
  dat2= proc_bipolarEMG(dat2);
  dat2.clab= strcat('x', dat2.clab);
%  dat= proc_appendChannels(dat1, dat2);
  dat= dat2;
  return;
end

if ~isempty(chanind(dat, 'EMGl')),
  warning('EMGl already bipolar');
else
  EMGl= chanind(dat, 'EMGlp', 'EMGln');
  if length(EMGl)~=2,
    warning('bipolar:EMGlMONO','monopolar EMGl channel(s) not found');
  else
    dat.x(:,EMGl(1),:)= dat.x(:,EMGl(1),:)-dat.x(:,EMGl(2),:);
    dat.x(:,EMGl(2),:)= [];
    dat.clab{EMGl(1)} = 'EMGl';
    dat.clab(EMGl(2)) = [];
  end
end

if ~isempty(chanind(dat, 'EMGr')),
  warning('EMGr already bipolar');
else
  EMGr= chanind(dat, 'EMGrp', 'EMGrn');
  if length(EMGr)~=2,
    warning('bipolar:EMGrMONO','monopolar EMGr channel(s) not found');
  else
    dat.x(:,EMGr(1),:)= dat.x(:,EMGr(1),:)-dat.x(:,EMGr(2),:);
    dat.x(:,EMGr(2),:)= [];
    dat.clab{EMGr(1)} = 'EMGr';
    dat.clab(EMGr(2)) = [];
  end
end

if ~isempty(chanind(dat, 'EMGf')),
  warning('EMGf already bipolar');
else
  EMGf= chanind(dat, 'EMGfp', 'EMGfn');
  if length(EMGf)~=2,
    warning('bipolar:EMGfMONO','monopolar EMGf channel(s) not found');
  else
    dat.x(:,EMGf(1),:)= dat.x(:,EMGf(1),:)-dat.x(:,EMGf(2),:);
    dat.x(:,EMGf(2),:)= [];
    dat.clab{EMGf(1)} = 'EMGf';
    dat.clab(EMGf(2)) = [];
  end
end
