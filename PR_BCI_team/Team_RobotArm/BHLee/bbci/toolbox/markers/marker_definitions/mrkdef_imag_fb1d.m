function mrk = mrkdef_imag_fb1d(Mrk, file, opt);


%% ----- extract marker information from 1D cursor control feedback -----

%% TODO: check whether also middle class was used.
%% See kraulems_analysis/Amin_05_09_07/prepare_data

global problem_marker_bit

Mrk2= readMarkerTable(file, 'raw');
fs = Mrk.fs;

if ~isfield(opt,'logf')
  [Mrk2,logf,flogf] = extract_logfiles(Mrk2,file,opt);
else
  logf = opt.logf;
  flogf = opt.flogf;
end

Mrk = Mrk2;
Mrk.pos = ceil(Mrk.pos/Mrk2.fs*fs);
Mrk.fs = fs;

classDef = {1,2,11,12,21,22,60; 
            opt.classes_fb{:}, ...
            ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
            ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}], ...
            'free'};
mrk= makeClassMarkers(Mrk, classDef,0,0);
if all(sum(mrk.y(7,:),2)==0)
  if problem_marker_bit
    classDef = {30,10,12,20,22,32;
                'indicate', ...
                ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
                ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}],'free'};
  else
    
    classDef = {31,11,12,21,22,32;
                'indicate', ...
                ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
                ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}],'free'};
  end
  
  mrk= makeClassMarkers(Mrk2, classDef,0,0);
  ind = find(mrk.y(1,:));
  cl1 = find(sum(mrk.y([2,5],:),1));
  cl2 = find(sum(mrk.y([3,4],:),1));
  for i = 1:length(ind)
    in1 = find(cl1>ind(i));
    in2 = find(cl2>ind(i));
    if isempty(in1) & isempty(in2)
      in = [];
    elseif isempty(in1)
      in = cl2(in2(1));
      cl = 2;
    elseif isempty(in2)
      in = cl1(in1(1));
      cl = 1;
    else
      if cl1(in1(1))<cl2(in2(1))
        in = cl1(in1(1));
        cl = 1;
      else
        in = cl2(in2(1));
        cl = 2;
      end
    end
    if isempty(in) | (i<length(ind) & ind(i+1)<in)
      mrk.y(1,ind(i)) = 0;
    else
      mrk.y(1,ind(i)) = cl;
    end
  end
  mrk.y = cat(1,mrk.y(1,:)==1,mrk.y(1,:)==2,mrk.y(2:end,:));
  mrk.className = {opt.classes_fb{:},mrk.className{2:end}};
  ind = find(sum(mrk.y,1)>0);
  mrk.pos = mrk.pos(ind);
  mrk.toe = mrk.toe(ind);
  mrk.y = mrk.y(:,ind);
  mrk.toe(find(mrk.y(1,:)))= 1;
  mrk.toe(find(mrk.y(2,:)))= 2;
  mrk.toe(find(mrk.y(7,:)))= 60;
      
  while sum(mrk.y(1:2,1))==0
    mrk.pos = mrk.pos(2:end);
    mrk.toe = mrk.toe(2:end);
    mrk.y = mrk.y(:,2:end);
  end
  mrk2 = mrk;
  mrk.fs = opt.fs;
  mrk.pos = round(mrk.pos/mrk2.fs*mrk.fs);

else
  mrk2= makeClassMarkers(Mrk2, classDef,0,0);

end

%  mrk.log = logf;

mrord = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses',opt.classes_fb),0,'remainclasses');
mrres = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','hit*','miss*'),0,'remainclasses');
mrfree = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','free'),0,'remainclasses');
mrord2 = mrk_sortChronologically(mrk_selectClasses(mrk2,'remainclasses', opt.classes_fb),0,'remainclasses');
mrres2 = mrk_sortChronologically(mrk_selectClasses(mrk2, 'remainclasses','hit*','miss*'),0,'remainclasses');

mrk.className = mrord.className;
mrk.pos = [];mrk.toe = [];mrk.y = [];
mrk.ishit = []; mrk.free = []; mrk.duration = [];

for i = 1:length(mrord.pos)
  if i==length(mrord.pos), 
    pos1 = inf; 
  else
    pos1 = mrord.pos(i+1);
  end
  ind = find(mrres.pos>mrord.pos(i)); 
  if isempty(ind), continue; end
  pos2 = ind(1);
  ind = find(mrfree.pos>mrord.pos(i)); 
  if isempty(ind), continue; end
  pos3 = ind(1);
  if mrfree.pos(pos3)<mrres.pos(pos2) & mrres.pos(pos2)<pos1 & ...
        (pos3==length(mrfree.pos) | mrfree.pos(pos3+1)>mrres.pos(pos2))
    mrk.pos = [mrk.pos, mrord.pos(i)];
    mrk.toe = [mrk.toe, mrord.toe(i)];
    mrk.y = [mrk.y,mrord.y(:,i)];
    mrk.ishit = [mrk.ishit, ([1, 1, 0 0]*mrres.y(:,pos2))>0];
    mrk.free = [mrk.free, mrfree.pos(pos3)];
    mrk.duration = [mrk.duration, mrres2.pos(pos2)-mrord2.pos(i)] ...
        / mrk2.fs*1000;
    
  end
end
mrk.indexedByEpochs = {'ishit','free','duration','run_no'};
mrk.run_no = zeros(1,length(mrk.pos));
if ~isempty(logf),
  for i = 1:size(logf.segments.ival,1)
    idx= find(mrk.pos>=logf.segments.ival(i,1) & ...
              mrk.pos<=logf.segments.ival(i,2));
    mrk.run_no(idx)= i;
  end
end


mrk.flogf = flogf;
mrk.logf = logf;
