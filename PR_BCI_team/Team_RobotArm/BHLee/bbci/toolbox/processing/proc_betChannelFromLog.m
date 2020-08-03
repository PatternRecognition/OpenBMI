function cnt= proc_betChannelFromLog(cnt, field)

ctrllog= eegfile_loadMatlab(cnt.file, 'vars','log');
fblog= eegfile_loadMatlab(cnt.file, 'vars','feedback');

fbchan= strukt('fs',cnt.fs, ...
               'clab',{field}, ...
               'x',NaN*zeros([size(cnt.x,1) 1]));
lag= min(diff(ctrllog.cls.pos));

xi_start= ceil(ctrllog.cls.pos(1)/4);
iCursor= find(fblog.update.object==1);
ct= NaN*zeros(ceil(fblog.update.pos(end)/4), 1);
for k= 1:length(iCursor),
  ii= iCursor(k);
  if strcmp(fblog.update.prop{ii}(1), field),
    xi= ceil(fblog.update.pos(ii)/4);
    xdata= fblog.update.prop_value{ii}(1);
    ct(xi)= xdata{1};
  end
end
%% match data segments
iSeg= find(fblog.mrk.toe==210);
eSeg= find(fblog.mrk.toe==212);
iSeg_ctrl= find(ctrllog.mrk.toe==210);
eSeg_ctrl= find(ctrllog.mrk.toe==212);
nSeg= length(iSeg);
if length(eSeg)>length(iSeg),
  warning('more segment end than segment start markers');
  toomuch= inf;
  while toomuch>1,
    toomuch= length(eSeg)-length(iSeg);
    dd= eSeg-iSeg([1:end end*ones(1,toomuch)]);
    idel= min(find(dd<0));
    if isempty(idel),
      idel= length(iSeg)+1;
    end
    eSeg(idel)= [];
  end
end
if length(eSeg_ctrl)>length(iSeg_ctrl),
  warning('more segment end than segment start markers');
  toomuch= inf;
  while toomuch>1,
    toomuch= length(eSeg_ctrl)-length(iSeg_ctrl);
    dd= eSeg_ctrl-iSeg_ctrl([1:end end*ones(1,toomuch)]);
    idel= min(find(dd<0));
    if isempty(idel),
      idel= length(iSeg_ctrl)+1;
    end
    eSeg_ctrl(idel)= [];
  end
end

if nSeg~=length(eSeg) | nSeg~=length(iSeg_ctrl) | nSeg~=length(eSeg_ctrl),
  error('inconsistent segment marker');
end
for si= 1:nSeg,
  ival_source= ceil([fblog.mrk.pos(iSeg(si)) fblog.mrk.pos(eSeg(si))]/4);
  ival_source= ival_source - xi_start + 1;
  ival_target= [ctrllog.mrk.pos(iSeg_ctrl(si)) 
                ctrllog.mrk.pos(eSeg_ctrl(si))];
  iv_source= [ival_source(1):ival_source(2)];
  iv_target= [ival_target(1):lag:ival_target(2)];
  if length(iv_source)>length(iv_target),
    warning('need to prune source ival');
    iv_source(end)= [];
  end
  if length(iv_source)<length(iv_target),
    warning('need to prune target ival');
    iv_target(end)= [];
  end
  for k= 0:lag-1,
    fbchan.x(iv_target+k,1)= ct(iv_source);
  end
end

cnt= proc_appendChannels(cnt, fbchan);
