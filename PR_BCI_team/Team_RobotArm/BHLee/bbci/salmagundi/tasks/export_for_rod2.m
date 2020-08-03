file= 'Matthias_06_02_09/imag_lettMatthias';
file_fb= 'Matthias_06_02_09/imag_fb1drnocursMatthias';
%file= 'Michael_06_07_10/imag_lettMichael';
%file_fb= 'Michael_06_07_10/imag_1drfbMichael';

%% get original parameter settings
bbci= eegfile_loadMatlab(file_fb, 'vars','bbci');
filt_b= bbci.analyze.csp_b;
filt_a= bbci.analyze.csp_a;
csp_w_orig= bbci.analyze.csp_w;
csp_clab= bbci.setup_opts.clab;

%% train classifier on calibration data
fv= copy_struct(bbci.analyze.features, 'x','y');
C= trainClassifier(fv, bbci.setup_opts.model);


winlen_list= [100 200 500 750 1000 1500];

[cnt, mrk, mnt]= eegfile_loadMatlab(file_fb);
[ctrllog]= eegfile_loadMatlab(cnt.file, 'vars','log');
clear cls
eval(ctrllog.changes.code{1})
if ~exist('cls','var'),
  cls.bias= 0;
  cls.scale= 0.1;
  cls.dist= 0;
  cls.alpha= 1;
  cls.range= [-1 1];
%  if strcmp(fileparts(file), 'Matthias_06_02_09'),
%    S= load(ctrllog.setup_file{3}); 
%    cls= S.cls;
%  else
%    error('define classifier setup');
%  end
end

cnt= proc_selectChannels(cnt, csp_clab);
cnt= proc_linearDerivation(cnt, csp_w_orig, 'prependix','csp');
cnt= proc_filt(cnt, filt_b, filt_a);

for ww= 1:length(winlen_list),
  ctrl= proc_movingVariance(cnt, winlen_list(ww));
  ctrl= proc_logarithm(ctrl);
  ctrl= proc_linearDerivation(ctrl, C.w);
  ctrl.clab= {sprintf('ctrl_%d', winlen_list(ww))};
  ctrl.x= ctrl.x + C.b;
  ctrl.x= (ctrl.x + cls.bias) * cls.scale;
  ctrl.x= sign(ctrl.x).*max(0,(abs(ctrl.x)-cls.dist)./(1-cls.dist));
  ctrl.x= sign(ctrl.x).*(abs(ctrl.x).^cls.alpha);
  ctrl.x= min(max(ctrl.x, cls.range(1)), cls.range(2));
  if ww==1,
    ctrlms= ctrl;
  else
    ctrlms= proc_appendChannels(ctrlms, ctrl);
  end
end
%ctrlms= proc_subsampleByLag(ctrlms, 4);
%mrk.pos= ceil(mrk.pos/4);
%mrk.free= ceil(mrk.free/4);
%mrk.fs= mrk.fs/4;

%% delete pauses at beginning and end
todel= [1:mrk.pos(1)-mrk.fs, ...
        mrk.pos(end)+round((mrk.duration(end)/1000+1)*mrk.fs):size(ctrlms.x,1)];
%% delete pauses inbetween
dp= diff(mrk.pos/mrk.fs) - mrk.duration(1:end-1)/1000;
ipause= find(dp>5);
for pp= 1:length(ipause),
  ip= ipause(pp);
  td= mrk.pos(ip)+round((mrk.duration(ip)/1000+2)*mrk.fs) : ...
      mrk.pos(ip+1)-2*mrk.fs;
  todel= [todel td];
end

%% save position control data
x= ctrlms.x;
x(todel,:)= [];
top= max(abs(percentiles(x(:,6), [2.5 97.5])));
x= x/top*0.95;
save([EEG_EXPORT_DIR 'MK060209_pc_multiscale.txt'], 'x', '-ASCII');


%% transform to rate control
rcms= zeros(size(ctrlms.x));
for ee= 1:length(mrk.pos),
  zz= zeros([1 length(winlen_list)]);
  iEnd= mrk.pos(ee) + ceil(mrk.duration(ee)/1000*mrk.fs);
  for nn= mrk.free(ee):iEnd,
    zz= zz + ctrlms.x(nn, :);
    rcms(nn,:)= zz;
  end
end

rcms(todel,:)= [];
top= max(abs(percentiles(rcms(:,6), [2.5 97.5])));
rcms= rcms/top*0.95;
save([EEG_EXPORT_DIR 'MK060209_rc_multiscale.txt'], 'rcms', '-ASCII');
