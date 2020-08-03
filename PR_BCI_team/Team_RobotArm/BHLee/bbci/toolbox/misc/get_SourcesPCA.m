function [SourceVectors, SourceWaveforms,values] = get_SourcesPCA(cnt,mrk,options, varargin);
%GET_MSEC
% cnt usual datas
% mrk markers
% options is a struct with the following entries
%    - same: flag if you want to handle each artifacts for itself(=0) (default) or in the big covariance together, then this field says how much source vectors you get.  
%    but note, you must have then equal time series!!!!! (if >0)
%    - lowpass: a range for lowpass filtering
% functions a cell array of functions for the preprocessing of the artifacts
% varargin is a cell array with structs (or only name) with the following entries
%    - name : name of the artifacts 
%    - artifacts: the artifacts (see mrk) who belongs to the artifact (default = name)
%    - range: time before and after onset (default: [-500 500])
%    - postpone: a range in which the maximum gradient are allowed to be and a postponement of the time series goes to this point (default 0)
%    - channels: a channel name where the gradient must be, or a cell array and factors of its mass or a number with masses for all channels you get from dat.clab.  (default the last with eqal priors), or for the last( identities))
%    - combination: if more than one artifact are given, (e. g. left and right and you are interested in horizontal movement, but you want to invert right first), you can give here masses of each artifact (in our example [1 -1]). (default [1 ...1]).
%    - number: the number of source vectors and source waveforms you want to get back from this artifact. (default 1)
%    - separations: in mrk.int you get intervals, if this real intervals, separation says how often you must have an artifact in this interval (default 1 (at left end, rest is equal partiotioned))
%    - belongto : if more than one artifact is given, this matrices
%    describes for each element of separation (in columns) and for
%    each artifact (in rows), where it belongs to in order fo time.
%
% output:
%    - SourceVectors: a big array with number channels rows (and columns sourcevectors)
%    - SourceWaveforms: Cell-Array, in each cell there are some source waveform
%    - values: a vector with the proportion of variance in the PCA
%
% Guido Dornhege
% 23.02.02

if ~exist('varargin') | isempty(varargin)
    error('where are the artifacts?');
end

if ~exist('options') | isempty(options)
    options.same = 0;
end
 
if ~exist('functions')
  functions = [];
end

if ~isstruct(options)
    options.same = options;
end

if ~isfield(options,'same')
  options.same = 0;
end

if ~isfield(options,'lowpass')
  options.lowpass = [];
end


for i = 1:length(varargin) 
    if ~isstruct(varargin{i}) 
        artifact(i).name = varargin{i};
    else
        artifact(i).name = varargin{i}.name;
    end
    if ~isfield(varargin{i},'artifacts') 
        artifact(i).artifacts ={artifact(i).name};
    else
        artifact(i).artifacts = varargin{i}.artifacts;
    end
    
    if ~iscell(artifact(i).artifacts)
        artifact(i).artifacts = {artifact(i).artifacts}; 
    end
 
   
    if ~isfield(varargin{i},'range')
        artifact(i).range = [-500 500];
    else
        artifact(i).range = varargin{i}.range;
    end
   
    if ~isfield(varargin{i},'postpone')
        artifact(i).postpone = [0 0];
    else
        artifact(i).postpone = varargin{i}.postpone;
    end

    if ~isfield(varargin{i},'channels')
        artifact(i).channels = ones(1,length(cnt.clab))/length(cnt.clab);
    elseif ischar(varargin{i}.channels)
        c = find(strcmp(cnt.clab,varargin{i}.channels));
        artifact(i).channels = zeros(1,length(cnt.clab));
        artifact(i).channels(c) = 1;
    elseif size(varargin{i}.channels,1) == 1
        artifact(i).channels = varargin{i}.channels;
    elseif size(varargin{i}.channels,1) == 2
        ch = cat(1,{varargin{i}.channels{1,:}});
        ch = intersect(ch,cnt.clab);
        artifact(i).channels = zeros(1,length(cnt.clab));
        for j = 1:length(ch)
            c = find(strcmp(ch{j},cnt.clab));
            artifact(i).channels(c) = 1/length(ch);
        end
        
    else
        error('Format of channels not supported');
    end
    
    if ~isfield(varargin{i},'combination')
        artifact(i).combination = ones(1, length(artifact(i).artifacts));
    else
        artifact(i).combination = varargin{i}.combination;
    end
    
    if ~isfield(varargin{i},'number')
        artifact(i).number = 1;
    else
        artifact(i).number = varargin{i}.number;
    end
    if ~isfield(varargin{i},'separations')
        artifact(i).separations = 1;
    else
        artifact(i).separations =  varargin{i}.separations;   
    end
    if ~isfield(varargin{i},'belongto')
        artifact(i).belongto = (1:length(artifact(i).artifacts))'*ones(1,artifact(i).separations);
    else
        artifact(i).belongto = varargin{i}.belongto;
    end
 
end


iv = get_relevantArtifacts(mrk,artifact(:).artifacts);


if options.same==0
    covar = zeros(length(cnt.clab),length(cnt.clab),length(iv.int));
else    
    covar = zeros(length(cnt.clab),length(cnt.clab));
end

if options.same> 0 datas = []; end
%Point of interests
for i = 1:length(iv.int)
    b = iv.int(i);
    c = b{:};
    sep = artifact(i).separations;
    points = repmat(c(:,1),1,sep);
    if sep ==1
        steps = 0;
    else
    steps = round((c(:,2)-c(:,1))./(sep-1));
end
    add = steps*(0:(sep-1));
    points = points +add;
    class = artifact(i).belongto(iv.cl{i},:); 
    points = points(1:prod(size(points)));
    class = class(1:prod(size(class)));
    [iv.points{i},ind] = union(points,[]);
    iv.class{i} = class(ind);
    if (artifact(i).postpone(2)-artifact(i).postpone(1))==0
        iv.points{i} =iv.points{i}+artifact(i).postpone(1)*cnt.fs/1000;
    else
        for j = 1:length(iv.points{i})
           dat = cnt.x(getfield(iv.points{i},{j})+(round(artifact(i).postpone(1)*cnt.fs/1000):round(artifact(i).postpone(2)*cnt.fs/1000)),:,:);
           dat = dat*artifact(i).channels';
           diff = dat(2:end,1)-dat(1:end-1,1);
           [dummy,place] = max(abs(diff));
           setfield(iv.points{i},{j},place+getfield(iv.points{i},{j})+artifact(i).postpone(1));
        end   
    end
    
    data = []; 
    for j = 1:length(iv.points{i})
       dat = cnt;
       dat.x = cnt.x(getfield(iv.points{i},{j})+ ...
		     (round(artifact(i).range(1)*cnt.fs/1000): ...
		      round(artifact(i).range(2)*cnt.fs/1000)),:,:);
       
       if ~isempty(options.lowpass)
        
	 dat = proc_filtBruteFFT(dat,options.lowpass, size(dat.x,1)*1000/cnt.fs, size(dat.x,1)*1000/cnt.fs);
	 
       end
       
        dat = dat.x*artifact(i).combination(getfield(iv.class{i},{j}));
        %data = cat(1,data,dat - repmat(mean(dat,1),size(dat,1),1)); 
        data = cat(3,data,dat);        
    end
    if options.same==0
        meandat{i} = mean(data,3);
        covar(:,:,i) = cov(meandat{i},1);
        %meandat{i} = mean(data2,3);
    else 
        datas = cat(3,datas,data);
        %datas2 = cat(3,datas2,data2);
    end
end
if options.same>0 
    mittDat = mean(datas,3);
    covar = cov(mittDat,1);
    [V,D] = eigs(covar,options.same,'LM',struct('disp',0));
    SourceVectors = V;
    values = 100*sum(diag(D))/trace(covar);
    SourceWaveforms = {pinv(SourceVectors)*mittDat'};
else
    SourceVectors = [];
    SourceWaveforms =[];
    for i = 1:length(iv.int)
        [V,D] = eigs(covar(:,:,i),artifact(i).number,'LM',struct('disp',0));
        SourceVectors = cat(2,SourceVectors,V);
        values(i) = 100*sum(diag(D))/trace(covar(:,:,i));
        SourceWaveforms = cat(1,SourceWaveforms,{pinv(V)*meandat{i}'});
    end
end    











