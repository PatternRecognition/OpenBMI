file= 'Guido_05_11_15/imag_1drfbGuido';
%file= 'John_05_11_15/imag_1drfbJohn1';
%file= 'John_05_11_15/imag_1drfbJohn2';

is= max(find(isletter(file) & file<'Z'));
sbj= file(is:end);

fb= eegfile_loadMatlab(file, 'vars','log');

xi= ceil(fb.cls.pos/4);
xi_start= xi(1);
xi= xi-xi_start+1;
N= xi(end);

cs= NaN*zeros(N, 1);
cs(xi)= fb.cls.values;
save([EEG_EXPORT_DIR 'for_john/' sbj '_cls'], '-ASCII','cs');

if ~isequal(fb.cls.pos, fb.udp.pos),
  error('mismatch');
end

cs(xi)= fb.udp.values;
save([EEG_EXPORT_DIR 'for_john/' sbj '_udp'], '-ASCII','cs');


cs= zeros(N, 1);
ii= find(ismember(fb.mrk.toe,[1 2 11 12 21 22 60]));
xi= ceil(fb.mrk.pos(ii)/4);
xi= xi-xi_start+1;
cs(xi)= fb.mrk.toe(ii);
save([EEG_EXPORT_DIR 'for_john/' sbj '_labels'], '-ASCII','cs');


ctrl= eegfile_loadMatlab(file, 'vars','feedback');
iCursor= find(ctrl.update.object==1);
cs= NaN*zeros(N, 1);
ct= NaN*zeros(ceil(ctrl.update.pos(end)/4), 1);
for k= 1:length(iCursor),
  ii= iCursor(k);
  if strcmp(ctrl.update.prop{ii}(1), 'XData'),
    xi= ceil(ctrl.update.pos(ii)/4);
    xdata= ctrl.update.prop_value{ii}(1);
    ct(xi)= xdata{1};
  end
end
%% match data segments
iSeg= find(ctrl.mrk.toe==210);
eSeg= find(ctrl.mrk.toe==212);
iSeg_fb= find(fb.mrk.toe==210);
eSeg_fb= find(fb.mrk.toe==212);
nSeg= length(iSeg);
if length(eSeg_fb)==nSeg+1 & abs(iSeg_fb(1)-eSeg_fb(1))<10,
  eSeg_fb(1)= [];
end
if nSeg~=length(eSeg) | nSeg~=length(iSeg_fb) | nSeg~=length(eSeg_fb),
  error('mismatch');
end
for si= 1:nSeg,
  ival_source= ceil([ctrl.mrk.pos(iSeg(si)) ctrl.mrk.pos(eSeg(si))]/4);
  ival_source= ival_source - xi_start + 1;
  ival_target= ceil([fb.mrk.pos(iSeg_fb(si)) fb.mrk.pos(eSeg_fb(si))]/4);
  iv_source= [ival_source(1):ival_source(2)];
  iv_target= [ival_target(1):ival_target(2)];
  if length(iv_source)>length(iv_target),
    iv_source(end)= [];
  end
  if length(iv_source)<length(iv_target),
    iv_target(end)= [];
  end
  cs(iv_target)= ct(iv_source);
end

save([EEG_EXPORT_DIR 'for_john/' sbj '_ctrl'], '-ASCII','cs');

%cs= zeros(ceil(ctrl.update.pos(end)/4), 1);
%ii= find(ismember(ctrl.mrk.toe,[1 2 11 12 21 22 60]));
%xi= ceil(ctrl.mrk.pos(ii)/4);
%cs(xi)= ctrl.mrk.toe(ii);
%save([EEG_EXPORT_DIR 'for_john/' sbj '_ctrl_labels'], '-ASCII','cs');












return

iv= 1:100;
while iv(end)<=length(cs),
  while iv(end)<=length(cs) & ~all(cs(iv)==0),
    iv= iv + 10;
  end
  if iv(end)<=length(cs),
    is= max([0; find(cs(1:iv(1))~=0)]) + 1;
    ie= iv(end) + min(find(cs(iv(end)+1:end)~=0));
    cs(is:ie)= NaN;
    iv= iv + ie - iv(1);
  end
end
invalid= find(isnan(cs));
cs(invalid)= [];
