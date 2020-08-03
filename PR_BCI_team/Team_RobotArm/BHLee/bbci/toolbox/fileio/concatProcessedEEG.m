function [Cnt, Mrk, mnt, N]= concatProcessedEEG(file_list, varargin)
%[Cnt, Mrk, mnt, N]= concatProcessedEEG(file_list, varargin)
%
% N - vector of number of events in each file

if nargin>1 & isstruct(varargin{1}),
  opt= varargin{1};
  param= varargin(2:end);
else
  opt= struct([]);
  param= varargin;
end

if ~iscell(file_list),
  file_list= {file_list};
end
N= zeros(1, length(file_list));
T= zeros(1, length(file_list));
for ii= 1:length(file_list),
  if iscell(file_list{ii})
    cnt = file_list{ii}{1};
    mrk = file_list{ii}{2};
    mnt = file_list{ii}{3};
  else
    [cnt, mrk, mnt]= loadProcessedEEG(file_list{ii}, param{:});
    if ~isstruct(cnt),  %% when variable 'cnt' is not to be loaded
      cnt= struct('x',[], 'clab',{{}}, 'fs',0, 'title','', 'file','');
    end
  end
  if isfield(opt, 'proc'),
    eval(opt(min(ii,length(opt))).proc);
  end
%% --- begin of hack
  if cnt.fs==0 & isstruct(mrk),  %% cnt was not loaded, but mrk
    [dmy,dmy,dmy,dmy,T_sec]= readGenericHeader(file_list{ii});
    T(ii)= round(T_sec*mrk.fs);
  else
%% --- end of hack
    T(ii)= size(cnt.x,1);
  end
  N(ii)= sum(any(mrk.y,1));
  if ii==1,
    Cnt= cnt;
    Mrk= mrk;
    if ndims(Cnt.x)==3,
      Mrk= rmfield(Mrk, 'pos');
%      if isfield(mrk, 'trg'),
%        Mrk= rmfield(Mrk, 'trg');
%      end
    end
  else
    if ~isequal(cnt.clab, Cnt.clab), warning('inconsistent clab structure will be repaired by using the intersection'); 
      chan = intersect(cnt.clab,Cnt.clab);
      cnt = proc_selectChannels(cnt,chan{:});
      Cnt = proc_selectChannels(Cnt,chan{:});
    end
    if ~isequal(cnt.fs, Cnt.fs), error('inconsistent sampling rate'); end
    if ndims(Cnt.x)==2,
      Cnt.x= cat(1, Cnt.x, cnt.x);
      Mrk.pos= [Mrk.pos, mrk.pos + sum(T(1:ii-1))];
      if isfield(Mrk, 'indexedByEpochs'),
        for Fld= Mrk.indexedByEpochs,
          fld= Fld{1};
          if ~isfield(mrk, fld),
            Mrk= rmfield(Mrk, fld);
            Mrk.indexedByEpochs= setdiff(Mrk.indexedByEpochs, fld);
            if isempty(Mrk.indexedByEpochs),
              Mrk= rmfield(Mrk, 'indexedByEpochs');
            end
            warning(sprintf('field <%s> not found in all files: deleted',fld));
            continue;
          else
            Mrk_fld= getfield(Mrk, fld);
            mrk_fld= getfield(mrk, fld);
            Mrk= setfield(Mrk, fld, cat(ndims(Mrk_fld), Mrk_fld, mrk_fld));
          end
        end
      end
    else           %% data are epoched: append trials
      if size(cnt.x,1)~=size(Cnt.x,1), error('inconsistent trial length'); end
      Cnt.x= cat(3, Cnt.x, cnt.x);
    end
    Mrk.toe= [Mrk.toe, mrk.toe];
    if isfield(Mrk, 'className'),
      Mrk.y= cat(2, Mrk.y, zeros(size(Mrk.y,1),size(mrk.y,2)));
      for i = 1:length(mrk.className)
        c = find(strcmp(Mrk.className,mrk.className{i}));
        if isempty(c)
          Mrk.y= cat(1, Mrk.y, zeros(1,size(Mrk.y,2)));
          Mrk.className=  cat(2, Mrk.className, {mrk.className{i}});
          c= size(Mrk.y,1);
        elseif length(c)>1,
          error('multiple classes have the same name');
        end
        Mrk.y(c,end-size(mrk.y,2)+1:end)= mrk.y(i,:);
      end
    else
      Mrk.y= cat(2, Mrk.y, mrk.y);
    end
  end
  if ii==1,
    Cnt.file= {};
  end
  Cnt.file= cat(2, Cnt.file, {cnt.file});
end
if ndims(Cnt.x)==2,
  Cnt.T= T;
end

if length(file_list)>1,
  Cnt.title= [Cnt.title ' et al.'];
end
