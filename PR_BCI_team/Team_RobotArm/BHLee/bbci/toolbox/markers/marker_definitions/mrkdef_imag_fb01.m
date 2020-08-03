function mrk = mrkdef_imag_fb01(Mrk, file, opt);


%% ----- extract marker information from 1D fixed duration
%% ----- initial adaptation cursor off feedback.

%% cursor on or off is indicated by opt.cursor_on.
opt.cursor_on = 0;
opt.adaptation = 1;
opt.markertypes = [1 2 11 12 21 22 60 70 71 110]
global problem_marker_bit

Mrk2= readMarkerTable(file, 'raw');
fs = Mrk.fs;
if ~isfield(opt,'logf')
  if strcmp(opt.marker_type,'response')
    % This is a hack and will most probably not work for 2 players.
    Mrk2.toe = abs(Mrk2.toe);
  end
  [Mrk2,logf,flogf] = extract_logfiles(Mrk2,file,opt);
else
  logf = opt.logf;
  flogf = opt.flogf;
end

% the run number is determined by the filename.
if file(end)>'1'&file(end)<='9'
  run_baseidx = 1+str2num(file(end));
else
  run_baseidx = 0;
end

Mrk = Mrk2;
Mrk.pos = ceil(Mrk.pos/Mrk2.fs*fs);
Mrk.fs = fs;

classDef = {1,2,11,12,21,22,60,70,71,110; 
            opt.classes_fb{:}, ...
            ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
            ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}], ...
            'free', 'rotation', 'rotation off','free_init'};
mrk= makeClassMarkers(Mrk, classDef,0,0);
mrk2= makeClassMarkers(Mrk2, classDef,0,0);

%  mrk.log = logf;

mrord = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses',opt.classes_fb),0,'remainclasses');
mrres = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','hit*','miss*'),0,'remainclasses');
mrfree = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','free*'),0,'remainclasses');
mrord2 = mrk_sortChronologically(mrk_selectClasses(mrk2,'remainclasses', opt.classes_fb),0,'remainclasses');
mrres2 = mrk_sortChronologically(mrk_selectClasses(mrk2, 'remainclasses','hit*','miss*'),0,'remainclasses');
mrrot = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','rotation*',0,'remainclasses'));


mrk.className = {mrord.className{:},'rotation'};
mrk.pos = [];mrk.toe = [];mrk.y = [];
mrk.ishit = []; mrk.free = []; mrk.duration = [];
mrk.adaptation_trial = [];

for ii = 1:2:length(mrrot.pos)
  % initial rotation phase.
  mrk.pos = [mrk.pos, mrrot.pos(ii)];
  mrk.toe = [mrk.toe, mrrot.toe(ii)];
  mrk.y = [mrk.y, [0;0;1]];
  mrk.ishit = [mrk.ishit, 0];
  mrk.free = [mrk.free, mrrot.pos(ii)];
  mrk.duration = [mrk.duration mrrot.pos(ii+1)-mrrot.pos(ii)];
  mrk.adaptation_trial = [mrk.adaptation_trial 1];
end

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
  % Distinguish initial trials and free gaming:
  init_ind = find(mrfree.y(:,pos3));
  
  if mrfree.pos(pos3)<mrres.pos(pos2) & mrres.pos(pos2)<pos1 & ...
        (pos3==length(mrfree.pos) | mrfree.pos(pos3+1)>mrres.pos(pos2))
    mrk.pos = [mrk.pos, mrord.pos(i)];
    mrk.toe = [mrk.toe, mrord.toe(i)];
    mrk.y = [mrk.y,[mrord.y(:,i);0]];
    mrk.ishit = [mrk.ishit, ([1, 1, 0 0]*mrres.y(:,pos2))>0];
    mrk.free = [mrk.free, mrfree.pos(pos3)];
    mrk.duration = [mrk.duration, mrres2.pos(pos2)-mrord2.pos(i)] ...
        / mrk2.fs*1000;
    if strcmp(mrfree.className{init_ind},'free')
      mrk.adaptation_trial = [mrk.adaptation_trial 0];
    else
      mrk.adaptation_trial = [mrk.adaptation_trial 1];
    end
  end
end
mrk.indexedByEpochs = {'ishit','free','duration','run_no','adaptation_trial','cursor_on'};


mrk.run_no = zeros(1,length(mrk.pos));
if ~isempty(logf),
  run_count = 0;
  for i = 1:size(logf.segments.ival,1)
    idx= find(mrk.pos>=logf.segments.ival(i,1) & ...
              mrk.pos<=logf.segments.ival(i,2));
    if ~isempty(idx)
      run_count = run_count+1;
      mrk.run_no(idx)= run_count;
    end
  end
end

mrk.run_no = mrk.run_no+run_baseidx;
mrk.cursor_on = ones(size(mrk.pos))*opt.cursor_on;

mrk = makeClassMarkers(mrk_sortChronologically(mrk),classDef);
%mrk = mrk_selectClasses(mrk,{opt.classes_fb{:},'rotation'});
mrk = mrk_selectClasses(mrk, opt.classes_fb);

mrk.flogf = flogf;
mrk.logf = logf;

