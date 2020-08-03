pydir= '/home/neuro/blanker/python/sandbox';

filelist= {...
    'Michael_06_07_10/imag_lettMichael', '', 'MS';
    'Matthias_06_02_09/imag_lettMatthias', 'imag_fb1drnocurs', 'MK';
    'John_05_11_15', '', 'JHW';
    'VPcm_06_02_21', '', ''; 
    'Maki_05_11_18', '', 'SF';
    'Gilles_05_11_16', '', 'GB';
    'Guido_05_11_15', '', 'GD'};

for ff= 1:size(filelist,1),
  
file= filelist{ff,1}
iu= min(find(file=='_'));
sbj= file(1:iu-1);
is= min(find(file=='/'));
if isempty(is),
  subdir= [file '/'];
  file= [subdir 'imag_lett' sbj]; 
else
  subdir= file(1:is);
end
dat= file(iu+1:is-1);
dat(dat=='_')= [];
if isempty(filelist{ff,2}),
  file_fb= [subdir 'imag_1drfb' sbj];
else
  file_fb= [subdir filelist{ff,2} sbj];
end
if isempty(filelist{ff,3}),
  if strcmp(sbj(1:2), 'VP'),
    tag= [sbj(3:4) dat];
  else
    tag= [sbj(1) dat];
  end
else
  tag= filelist{ff,3};
end

%% get original parameter settings
bbci= eegfile_loadMatlab(file_fb, 'vars','bbci');
filt_b= bbci.analyze.csp_b;
filt_a= bbci.analyze.csp_a;
csp_w_orig= bbci.analyze.csp_w;
csp_clab= bbci.setup_opts.clab;

%% train classifier on calibration data
fv= copy_struct(bbci.analyze.features, 'x','y');
C= trainClassifier(fv, bbci.setup_opts.model);


winlen_list= 100:100:1500;

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
  ctrlvar= proc_subtractMovingAverage(ctrl, 750, 'centered');
  ctrlvar= proc_movingVariance(ctrlvar, winlen_list(ww));
  ctrlvar= proc_movingAverage(ctrlvar, 500, 'centered');
  if ww==1,
    ctrlms= ctrl;
    pcvar= ctrlvar;
  else
    ctrlms= proc_appendChannels(ctrlms, ctrl);
    pcvar= proc_appendChannels(pcvar, ctrlvar);
  end
end
ctrlms= proc_subsampleByLag(ctrlms, 4);
pcvar= proc_subsampleByMean(pcvar, 4);
mrk.pos= ceil(mrk.pos/4);
mrk.free= ceil(mrk.free/4);
mrk.fs= mrk.fs/4;

if strcmp(tag, 'GB'),
  ii= find(ismember(mrk.run_no, [3 4 5]));
  mrk= mrk_selectEvents(mrk, ii)
end

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

pcvar_x= pcvar.x;
TT= size(pcvar_x,1);
if size(rcms,1)>TT,
  rcms= rcms(1:TT,:);
end
todel(find(todel>TT))= [];
rcms(todel,:)= [];
pcvar_x(todel,:)= [];
top= max(abs(percentiles(rcms(:,6), [2.5 97.5])));
rcms= rcms/top*0.95;
x= cat(2, rcms, pcvar_x);
save([EEG_EXPORT_DIR tag '_mscale_unc.txt'], 'x', '-ASCII');

cmd= sprintf('cd %s; python convert_multiscale_data.py %s mscale_unc', ...
             pydir, tag);
[s,w]= unix(cmd);
if s,
  fprintf('error: %s', w);
end
fprintf('%s: %g\n', tag, mean(rcms(:,end)));

end
