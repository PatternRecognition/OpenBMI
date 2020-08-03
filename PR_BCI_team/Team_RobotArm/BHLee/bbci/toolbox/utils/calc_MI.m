function MI = calc_MI(cnt, mrk, xTrials, range, varargin)
% cnt data, .proc preprocessing, .classy: to used classifier
% if you want to have preprocessings depending on labels, you have
% to set .proclabel to 1.
% mrk mrk structure
% xTrials: empty means test=trainset
%          one number divides in one trainset and tesatset to the
%          given ratio and runs one time
%          two number: "crossvalidation"
% range: time onset points 
% varargin: {1} interval for finding the classifier
%           .. for more arguments(jittering) by the call of
%           makeSegments
% MI: mutual information depend on time (see Schloegl at all,
% Estimating the Mutual Information ...)
%
% GUido Dornhege
%

if isfield(cnt, 'proc'),
  if exist(cnt.proc, 'file'),
    if isfield(cnt, 'proc_no'),
      proc_no= cnt.proc_no;
    else
      proc_no= 1;
    end
    proc= getBlockFromTape(cnt.proc, proc_no);
  else
    proc= cnt.proc;
  end
  proc= [proc ', fv= proc_flaten(fv);'];
else
  proc= 'fv= proc_flaten(fv);';
end

if ~isfield(cnt,'proclabel')
  cnt.proclabel = 0;
end

if ~isfield(cnt,'classy')
   cnt.classy = 'LDA';
end


[func, params]= getFuncParam(cnt.classy);
trainFcn= ['train_' func];
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end

if ~exist('xTrials') | isempty(xTrials)
   divTr{1}{1} = 1:length(mrk.y);
   divTe{1}{1} = 1:length(mrk.y);
elseif length(xTrials)==1
  [divTr,divTe] = sampleDivisions(mrk.y,[1 xTrials]);
  divTr{1} = {divTr{1}{1}};
  divTe{1} = {divTe{1}{1}};
else
   [divTr,divTe] = sampleDivisions(mrk.y,xTrials);
end

epo = makeSegments(cnt,mrk, varargin{:});
if cnt.proclabel
  y_memo = epo.y;
else
  eval(proc);
end

for p = 1: length(divTr)
  for q = 1:length(divTr{p})
    idxTr = divTr{p}{q};
    idxTe = divTe{p}{q};
    if cnt.proclabel
      epo.y(:,idxTe) = 0;
      eval(proc);
      epo.y = y_memo;
    end
    C{p,q} = feval(trainFcn, fv.x(:,idxTr), epo.y(:,idxTr),params{:}) ;
  end
end

for t = 1:length(range)
  epo = makeSegments(cnt,mrk, varargin{1}+range(t),varargin{2: ...
		    end});
  if ~cnt.proclabel  
    eval(proc);
  end
  
  for p = 1: length(divTr)
    for q = 1:length(divTr{p})
      idxTe = divTe{p}{q};
      if cnt.proclabel
	epo.y(idxTe) = 0;
	eval(proc);
	epo.y = y_memo;
      end
      
      outs{t,p,q} = feval(applyFcn, C{p,q}, fv.x(:,idxTe));
    end
  end
end

MI=[];
for p = 1: length(divTr)
  for q = 1:length(divTr{p})
    out = cat(1,outs{:,p,q});
    vages = var(out');
    idxTe = divTe{p}{q};
    valeft = var(transpose(out(:,find(fv.y(1,idxTe)))));
    varight = var(transpose(out(:,find(fv.y(2,idxTe)))));
    SNR = 2*vages./(valeft+varight)-1;
    MI = cat(1,MI,0.5*log2(1+SNR));
  end
end

if size(MI,1)>1
  MI = mean(MI);
end

             













