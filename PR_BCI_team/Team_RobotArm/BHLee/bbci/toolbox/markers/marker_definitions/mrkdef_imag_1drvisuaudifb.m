function mrk = mrkdef_imag_1drvisuaudifb(Mrk, file, opt);


%% ----- extract marker information from 1D cursor control feedback -----

Mrk2= readMarkerTable(file, 'raw', opt.marker_type);
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

classDef = {1,2,11,12,21,22,60,36;
            opt.classes_fb{:}, ...
            ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
            ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}], ...
            'free','state switch'};
mrk= makeClassMarkers(Mrk, classDef,0,0);
mrk2= makeClassMarkers(Mrk2, classDef,0,0);


mrord = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses',opt.classes_fb),0,'remainclasses');
mrres = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','hit*','miss*'),0,'remainclasses');
mrfree = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','free'),0,'remainclasses');
mrord2 = mrk_sortChronologically(mrk_selectClasses(mrk2,'remainclasses', opt.classes_fb),0,'remainclasses');
mrres2 = mrk_sortChronologically(mrk_selectClasses(mrk2, 'remainclasses','hit*','miss*'),0,'remainclasses');
mrswitch= mrk_selectClasses(mrk, 'state switch');

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
mrk.state_switch= mrswitch;
mrk.state= zeros(size(mrk.pos));
i0= 0;
state= 0;
for k= 1:length(mrswitch.pos),
  i1= max(find(mrk.pos<mrswitch.pos(k)));
  if isempty(i1), continue; end
  mrk.state(i0+1:i1)= state;
  i0= i1;
  state= 1-state;
end

mrk.indexedByEpochs = {'ishit','free','duration','run_no','state'};
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
