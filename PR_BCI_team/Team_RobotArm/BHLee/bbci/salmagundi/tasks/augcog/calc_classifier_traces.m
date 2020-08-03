clear fv classifier

di = '/home/schlauch/dornhege/augcog_season4/';
step = 200;
file = 'um_augcog_1';

fid = fopen([di file '.vmrk'],'r');

s = '';

while isempty(strmatch('Mk1=',s))
  s = fgets(fid);
end

fclose(fid);


c = strfind(s,',');
s = s(c(end)+1:end);

ye = str2num(s(1:4));
mo = str2num(s(5:6));
da = str2num(s(7:8));
ho = str2num(s(9:10));
mi = str2num(s(11:12));
se = str2num(s(13:14));

opt.sampling_fs = 100;

writelogfile(0,sprintf('classifier_sim_%sT%s.log',s(1:8),s(9:14)),opt.sampling_fs);
hyst_trans = [ 0, 0, 1 ];


S = load([di file(1:2) '_calc_classifier.cl'],'-mat');
classifier{1}= getfromdouble(S.classifier);
S = load([di file(1:2) '_audio_classifier.cl'],'-mat');
classifier{2}= getfromdouble(S.classifier);

out = zeros(1,length(classifier));          % classifier outputs
load_out = zeros(1,length(classifier));     % thresholded load outputs
state_filt = cell(1,length(classifier));    % filter states for online filtering
old_load_out = ones(1,length(classifier));  % old load outputs



cnt = readGenericEEG([di file]);
mrk = readMarkerTable([di file]);

step = step/1000*opt.sampling_fs;


for i = 1:length(classifier)
    % channel select
    classifier{i}.chan_sel = chanind(cnt,classifier{i}.clab);
    % band-pass filter
    [classifier{i}.filt_b,classifier{i}.filt_a] = butter(classifier{i}.filtOrder,classifier{i}.band/opt.sampling_fs*2);
    % setup parts of the fv struct which do not change
    fv(i) = struct('fs', opt.sampling_fs, ...
                   't', -classifier{i}.ilen_apply+1000/opt.sampling_fs ,...
                   'x', zeros(classifier{i}.ilen_apply/1000*opt.sampling_fs, length(classifier{i}.chan_sel)), ...
                   'clab', {cnt.clab(classifier{i}.chan_sel)} );
    % previous classifier outputs used for smoothing               
    out_old{i} = [];
    % spatial filtering (also detect if this fv.clab or fv.x
    [dum,state_spat{i}] = proc_spatial(classifier{i}.spatial.fcn, fv(i), classifier{i}.spatial.param{:});
    fv(i).x = zeros(classifier{i}.ilen_apply/1000*opt.sampling_fs, size(dum.x,2));
    fv(i).clab = dum.clab;
    
    % processing function for continuous data
    if isfield(classifier{i},'proc_cnt')
        for k = 1:length(classifier{i}.proc_cnt)
            state_cnt{i}{k} = [];
            classifier{i}.proc_cnt(k).fcn = lower(classifier{i}.proc_cnt(k).fcn);
        end
    end

    % classsifier output smoothing?
    if ~isfield(classifier{i},'integrate') | isempty(classifier{i}.integrate)
        classifier{i}.integrate = 1;
    end
    
    % clean up apply-evals (convert to lower case)
    for k = 1:length(classifier{i}.proc_apply)
        classifier{i}.proc_apply(k).fcn = lower(classifier{i}.proc_apply(k).fcn);
    end
%    classifier{i}.apply = lower(classifier{i}.apply);
end



pos = 0;
for i = step:step:size(cnt.x,1);
  fprintf('\r%i/%i     ',i,size(cnt.x,1));
  dat = cnt.x(pos+1:i,:);
  ind = find(mrk.pos>pos & mrk.pos<=i);
  for j = ind
    if mrk.toe(j)>0, desc = 'Stimulus';else desc = 'Response';end
    writelogfile(2,sprintf('Got marker: %s, Time: %i, Token: %i',desc,mrk.pos(j)*1000/opt.sampling_fs,abs(mrk.toe(j))));
  end
  for j = 1:length(classifier)
    
    
    C = classifier{j};
    
    % SELECT CHANNELS
    dat2 = dat(:, C.chan_sel);

        % FILT DATA 
        [dat2, state_filt{j}] = filter(C.filt_b, C.filt_a, dat2, state_filt{j});
    
         % SPATIAL FILTER
        dat2 = struct('fs', opt.sampling_fs, ...
                      'x', dat2, ...
                      'clab', {cnt.clab(C.chan_sel)});
        [dat2, state_spat{j}] = proc_spatial(C.spatial.fcn, dat2, state_spat{j});
        
        % further cnt_proc
        if isfield(C, 'proc_cnt')
            for k = 1:length(C.proc_cnt)
                [dat2, state_cnt{j}{k}] = feval(C.proc_cnt(k).fcn, ...
                                                dat2, state_cnt{j}{k}, C.proc_cnt(k).param{:});
            end
        end 
        
        %WINDOWING
        fv(j).x = cat(1, fv(j).x(size(dat2.x, 1)+1:end, :), dat2.x);
        if size(fv(j).x, 1) > C.ilen_apply/1000*opt.sampling_fs
            fv(j).x = fv(j).x(end - C.ilen_apply/1000*opt.sampling_fs+1:end, :);
        end

        % PROCESSING
        f = fv(j);
        for k = 1:length(C.proc_apply)
            f = feval(C.proc_apply(k).fcn, f, C.proc_apply(k).param{:});
        end
    
        % FEATURE VECTOR
        x = f.x(:);    

        %CLASSIFICATION
        outt = feval(C.apply, C.C, x);

        %SMOOTHING
        out_old{j} = [out_old{j}(max(1, length(out_old{j})-C.integrate+2):end), outt];
        out(j) = outt;
        
        %WORKLOAD
        load_out(j) = sum(mean(out_old{j})>C.mapping)+1;
        
        % do hysteresis mapping
        if isfield(C, 'hysterese') && C.hysterese == 1
            if load_out(j) == 2,
                load_out(j) = old_load_out(j);
            end

            old_load_out(j) = load_out(j);
        end
    end
    pos = i;
    
    
    
    for j = 1:length(classifier)
      if isfield(classifier{j}, 'hysterese') && classifier{j}.hysterese == 1
        load_out(j) = hyst_trans(load_out(j));
      end 
    end
    writelogfile(4,out,i*1000/opt.sampling_fs);
    writelogfile(3,load_out,i*1000/opt.sampling_fs);
    
    
    
    
    
end


writelogfile(1);
