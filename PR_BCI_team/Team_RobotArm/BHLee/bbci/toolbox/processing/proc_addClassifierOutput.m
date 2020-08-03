function cnt= proc_addClassifierOutput(cnt, method, varargin)
%cnt= proc_addClassifierOutput(cnt, szenario)
%cnt= proc_addClassifierOutput(cnt, method)

global EEG_RAW_DIR

[sub_dir, file]= fileparts(cnt.title);

if ischar(method),
  szenario= method;
  S= load([EEG_RAW_DIR sub_dir '/' szenario]);
  method= S.dscr;
  opt= S.opt;
else
  opt.feedback_step= 4;
  if ~isfield(method, 'scale'),
    method.scale= 1;
  end
  if ~isfield(method, 'proc'),
    method.proc= '';
  end
  if ~isfield(method, 'proc_apply'),
    method.proc_apply= method.proc;
  end
  if ~isfield(method, 'ilen_apply'),
    method.ilen_apply= method.ilen;
  end
  if ~isfield(method, 'iv_apply'),
    method.iv_apply= getIvalIndices([-method.ilen_apply 0], cnt.fs);
  end
end
for ii= 1:length(varargin),
  eval(sprintf('%s;', varargin{ii}));
end

fb= proc_selectChannels(cnt, method.clab);
if isfield(method, 'proc_cnt_apply') & ~isempty(method.proc_cnt_apply.fcn),
  fb= feval(method.proc_cnt_apply.fcn, fb, [], method.proc_cnt_apply.param{:});
end

iv= method.iv_apply - method.iv_apply(1) + 1;
epo= struct('fs', fb.fs);
method.applyFcn= getApplyFuncName(method.model);

fb_start= iv(end);
T= size(cnt.x,1);
t= floor((T-fb_start+1)/opt.feedback_step);
for ptr= 1:t,
  epo.x= fb.x(iv,:);
  fv= proc_applyProc(epo, method.proc_apply);
  oo= feval(method.applyFcn, method.C, fv.x(:));
  if ptr==1,
    nOut= size(oo,1);
    out= zeros(t, nOut);
  end
  out(ptr,:)= (method.scale .* oo)';
  iv= iv + opt.feedback_step;
end

if isfield(method, 'combinerFcn'),
  out= method.combinerFcn(out')';
  nOut= size(out,2);
end

iv= repmat(1:t, [opt.feedback_step 1]);
nChans= size(cnt.x,2);
cnt.x(:,nChans+[1:nOut])= NaN*zeros(T, nOut);
cnt.x(fb_start:fb_start+opt.feedback_step*t-1,nChans+[1:nOut])= ...
    out(iv(:),:);
cnt.clab= cat(2, cnt.clab, repmat({'out'}, [1 nOut]));
