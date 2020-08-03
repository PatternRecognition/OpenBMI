function [fv,apply] = get_augcog_csp(blk,varargin);
%GET_AUGCOG_CSP OPENS AN AUGCOG FILE AND CALCULATES CSP ALGORITHM
%
% usage:
%     fv = get_augcog_spectrum(file,<task=all available,ivallen=[1000 30000],band=[7 30],nPat=1,normalize='',channels={'not','E*','M*'});
%     fv = get_augcog_spectrum(blk,<task=all available,ivallen=[1000 30000],band=[7 30],nPat=1,normalize='',channels={'not','E*','M*'});
%     fv = get_augcog_spectrum(cnt,mrk,<ivallen=[1000,30000],band=[7 30],nPat=1,normalize='',channels={'not','E*','M*'});
%
% input:
%     file     -  the name of an augcog file
%     blk      -  a usual augcog blk structure
%     cnt,mrk  -  as usual
%     task     -  the task of the subject, e.g. '*auditory' 
%                 (for more than one as a cell array)
%     ivallen  -  the length of the ival (to make epochs)
%     band     -  the band spectral information should be given back, can have more than one row.
%     nPat     -  the number of used CSP Patterns
%                or {nPat,flag}, where flag is true, if CSP is really done, otherwise CSP is only prepared for classification.
%     normalize - a name of a normalization (or empty)
%
%
% output:
%     fv       - the resulting feature vector as a struct with fields:
%                .x    a spec*channel*trials matrix
%                .y    the logical 2xtrials matrix
%                .className  = {'low','high'};
%                .t    the time peaks
%                .clab a cell array consisting of the channel names
%                .task a logical array to divide into different sessions
%                .taskname a cell array of task names concerning .task
%                .proc if csp is only prepared
% Guido Dornhege, 27/04/2004


% load the file, if necessary
if ~isstruct(blk)
  blk = getAugCogBlocks(blk);
end

% set defaults
opt = setdefault(varargin,{'task','ivallen','band','nPat','normalize','channels'},...
                          {[],[1000,30000],[3 40],{1,false},'',{'not','E*','M*'}});

if length(opt.ivallen) == 1
  opt.ivallen = [1,1]*opt.ivallen;
end

% Choose the task
if ~isempty(opt.task) & ~isstruct(opt.task)
  if ~iscell(opt.task), opt.task = {opt.task};end
  blk = blk_selectBlocks(blk,opt.task{:});
end

% read the blocks
if ~isstruct(opt.task)
  [cnt,mrk] = readBlocks(blk);
else
  cnt = blk;
  mrk = opt.task;
end

% choose channels
if ~isempty(opt.channels)
  apply.chans = chanind(cnt,opt.channels{:});
  cnt = proc_selectChannels(cnt,opt.channels{:});
end

nchan = size(cnt.x,2);
para = {};
% band pass filtering
for i = 1:size(opt.band,1)
  [b,a] = butter(5,opt.band/cnt.fs*2);
  cc = proc_filt(cnt,b,a);
  if i ==1
    ccc = cc;
  else
    ccc.x = cat(2,ccc.x,cc.x);
  end
  para = {para{:},b,a};
end

apply.cnt_apply.fcn = 'proc_filtconcat';
apply.cnt_apply.param = para;

cnt = ccc;

% epoching
mrk = separate_markers(mrk);
mrk = mrk_setMarkers(mrk,opt.ivallen);

epo = makeEpochs(cnt,mrk,[0 opt.ivallen(2)]);

clear cnt mrk blk

make_proc_apply('init')
if ~isempty(opt.normalize)
  if iscell(opt.normalize)
    epo = feval(opt.normalize{1},epo,opt.normalize{2:end});
    str = sprintf('%s,',opt.normalize{2:end});
    str = str(1:end-1);
    make_proc_apply(sprintf('fv=%s(epo,%s);',opt.normalize{1},str));
  else
    epo = feval(opt.normalize,epo);
    make_proc_apply(sprintf('fv=%s(epo);',opt.normalize));
  end
end

epo.task = epo.y; epo.taskname = epo.className;
epo.indexedByEpochs = {'task','bidx'};
clInd0 = getClassIndices(epo,'base*');
clInd = getClassIndices(epo,'low*');
clInd2 = getClassIndices(epo,'high*');

epo.y = zeros(3,size(epo.y,2));
epo.className = {'base','low','high'};

ind0 = find(sum(epo.task(clInd0,:),1));
ind = find(sum(epo.task(clInd,:),1));
ind2 = find(sum(epo.task(clInd2,:),1));

epo.y(1,ind0) = 1;
epo.y(2,ind) = 1;
epo.y(3,ind2) = 1;

indi = find(sum(epo.y,2)>0);
epo.y = epo.y(indi,:);
epo.className = epo.className(indi);

epo.bidx = (1:size(epo.task,1))*epo.task;


% CSP
if iscell(opt.nPat) & length(opt.nPat)>1 & opt.nPat{2}==false
  epo.proc = sprintf('hlp_w=[];for hlp_ii = 1:%i;f = epo; f.x = f.x(:,(hlp_ii-1)*%i+1:hlp_ii*%i,:);[f,hlp_ww]=proc_csp(f,%i);if hlp_ii==1; fv=f; else; fv.x=cat(2,fv.x,f.x);end;hlp_w=[hlp_w,hlp_ww];end;fv=proc_logarithm(proc_variance(fv));hlp_classifier_rel=hlp_w;',size(opt.band,1),nchan,nchan,opt.nPat{1});
  str = sprintf('fv=epo;fv.x=fv.x(:,%i:%i,:);fv=proc_linearDerivation(fv,C(classifier).relevant_processing(:,1:%i));',1,nchan,opt.nPat{1}*2);
  for jjj = 2:size(opt.band,1)
    str = sprintf('%s hlp_fv2=epo;hlp_fv2.x=hlp_fv2.x(:,%i:%i,:);hlp_fv2=proc_linearDerivation(fv,C(classifier).relevant_processing(:,%i:%i));fv.x = cat(2,fv.x,hlp_fv2.x)',str,(hlp_ii-1)*nchan+1,hlp_ii*nchan,(hlp_ii-1)*opt.nPat{1}*2+1,hlp_ii*opt.nPat{1}*2);
  end
else
  if iscell(opt.nPat)
    opt.nPat = opt.nPat{1};
  end
  for hlp_ii = 1:size(opt.band,1);
    f = epo; f.x = f.x(:,(hlp_ii-1)*nchan+1:hlp_ii*nchan,:);
    [f,ww] = proc_csp(f,opt.nPat);
    if hlp_ii==1; fv=f; w=ww;
      str = sprintf('fv=epo;fv.x=fv.x(:,%i:%i,:);fv=proc_linearDerivation(fv,apply.proc_param(:,%i:%i));',1,nchan,1,opt.nPat*2);
    else; fv.x=cat(2,fv.x,f.x);w=[w,ww];
      str = sprintf('%s hlp_fv2=epo;hlp_fv2.x=hlp_fv2.x(:,%i:%i,:);hlp_fv2=proc_linearDerivation(hlp_fv2,apply.proc_param(:,%i:%i));fv.x = cat(2,fv.x,hlp_fv2.x);',str,(hlp_ii-1)*nchan+1,hlp_ii*nchan,(hlp_ii-1)*opt.nPat*2+1,hlp_ii*opt.nPat*2);

    end
  end
  epo = proc_logarithm(proc_variance(fv));
  apply.proc_param = ww;
end


make_proc_apply(str);
make_proc_apply('fv=proc_logarithm(proc_variance(epo));');


apply.proc_apply = make_proc_apply('get');


   

fv = epo;

