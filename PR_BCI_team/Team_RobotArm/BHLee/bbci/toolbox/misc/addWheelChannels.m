function cnt= addWheelChannels(cnt, factor)
%cnt= addWheelChannels(cnt, <factor=0.1>)

if ~exist('factor', 'var'), factor=0.1; end

mrk= readMarkerTable(cnt.title);
[sub_dir, file]= fileparts(cnt.title);
if sub_dir(1)~=filesep,
  global EEG_RAW_DIR
  sub_dir= [EEG_RAW_DIR sub_dir];
end

wheel_file= [sub_dir '/' file '_wheel.dat'];
steer= load(wheel_file);

iMrk= find(mrk.toe==-1);
iWheel= find(steer(:,4));
if length(iWheel)>length(iMrk),
  warning('inconsistent wheel markers, skipping last part');
  iWheel= iWheel(1:length(iMrk));
elseif length(iWheel)<length(iMrk),
  error('inconsistent wheel markers');
end

T= size(cnt.x,1);
t0= mrk.pos(iMrk(1));
t1= mrk.pos(iMrk(end));
xx= zeros(T, 2);
xx(1:t0,:)= repmat(steer(iWheel(1),2:3), [t0, 1]);
xx(t1:T,:)= repmat(steer(iWheel(end),2:3), [T-t1+1, 1]);

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

cnt.x= cat(2, cnt.x, xx*factor);
cnt.clab= cat(2, cnt.clab, {'wheel', 'throttle'});
