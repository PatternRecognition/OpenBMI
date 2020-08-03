function cnt= addSegmentedWheelChannels(cnt, factor)
%cnt= addSegmentedWheelChannels(cnt, <factor=0.1>)

if ~exist('factor', 'var'), factor=0.1; end
if ischar(cnt),
  cnt.title= cnt;
end

mrk= readMarkerTable(cnt.title);
[sub_dir, file]= fileparts(cnt.title);
if sub_dir(1)~=filesep,
  global EEG_RAW_DIR
  sub_dir= [EEG_RAW_DIR sub_dir];
end

brd= readSegmentBorders(cnt.title);
nSeg= size(brd.ival, 1);
xx= zeros(brd.ival(end), 2);
if ~isfield(cnt, 'x'),
  cnt.clab= {};
  cnt.fs= brd.fs;
  cnt.x= zeros(brd.ival(end), 0);
end
ptr= 0;
for is= 1:nSeg,
  wheel_file= [sub_dir '/' file '_wheel' int2str(is) '.dat'];
  steer= load(wheel_file);

  iMrk= find(mrk.toe==-1 & ...
             mrk.pos>=brd.ival(is,1) & mrk.pos<=brd.ival(is,2));
  iWheel= find(steer(:,4));
  if length(iWheel)>length(iMrk),
    warning(sprintf(['segment %d: ' ...
                     'inconsistent wheel markers, skipping last part'], is));
    iWheel= iWheel(1:length(iMrk));
  elseif length(iWheel)<length(iMrk),
    error('inconsistent wheel markers');
  end

  T= diff(brd.ival(is,:));
  t0= mrk.pos(iMrk(1)) - ptr;
  t1= mrk.pos(iMrk(end)) - ptr;
  xx(ptr+[1:t0],:)= repmat(steer(iWheel(1),2:3), [t0, 1]);
  xx(ptr+[t1:T],:)= repmat(steer(iWheel(end),2:3), [T-t1+1, 1]);
  ptr= brd.ival(is, 2);
  
  for ii= 1:length(iMrk)-1,
    ivw= iWheel(ii):iWheel(ii+1);
    ivm= mrk.pos(iMrk(ii)):mrk.pos(iMrk(ii+1));
    if length(ivw)==length(ivm),
      xx(ivm,:)= steer(ivw, 2:3);
    else
      xw= steer(ivw, 2:3);
      xm= resample(xw, length(ivm), length(ivw), 2);
      xx(ivm(1:end-1),:)= xm(1:end-1,:);
    end
  end
end

cnt.x= cat(2, cnt.x, xx*factor);
cnt.clab= cat(2, cnt.clab, {'wheel', 'throttle'});
