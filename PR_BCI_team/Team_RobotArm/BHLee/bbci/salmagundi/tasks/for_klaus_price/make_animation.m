%file= 'Guido_05_11_15/imag_lettGuido';
%file= 'Michael_05_11_10/imag_lettMichael';
%file= 'VPct_05_11_18/imag_lettVPct';
file= 'VPcm_06_02_07/imag_lettVPcm';

save_dir= [BCI_DIR 'tasks/for_klaus_price/'];

video= set_defaults([], ...
                    'size',[1024 768], ...
                    'fps', 25, ...
                    'duration', 60, ...
                    'save', 1, ...
                    'save_dir', [DATA_DIR 'eegVideo/tmp/'], ...
                    'save_name', 'csp_animation_%s.avi', ...
                    'maxFileSize',1000000000*2,...
                    'compress', 1, ...
                    'codec_spec', '-use_rgb -z -y xvid -R3 -w1');

anim= set_defaults([], ...
                   'eeg_clab', {'Fz','C3,4','P3,4'}, ...
                   'eeg_band', [3 45], ...
                   'eeg_scale', [], ...
                   'eeg_trace_color', [0 0 1], ...
                   'eeg_outline_color', [0 0 0], ...
                   'eeg_colrng', [0.5 4], ...
                   'window_length', [], ...
                   'video_start', [], ...
                   'show_class_tags', 1, ...
                   'tag_names', '%s imagery', ...
                   'tag_fontspec', {'FontSize',18}, ...
                   'csp1_ylim', [], ...
                   'csp2_ylim', [], ...
                   'csp1_trace_color', [0.85 0 0], ...
                   'csp1_outline_color', [1 0 0], ...
                   'csp2_trace_color', [0 0.65 0], ...
                   'csp2_outline_color', [0 0.7 0], ...
                   'csp_outline_width', 2, ...
                   'shade_ival', [750 3750], ...
                   'shade_color', [1 .8 .8; .8 1 .8; .8 .8 .8], ...
                   'draw_filters', 1, ...
                   'speed_factor', 1, ...
                   'ival_scalp', [2 3], ...
                   'indicate_ival', 1);
%                   'window_length', 10, ...
                   
opt_scalp= strukt('resolution', 50, ...
                  'shading','interp', ...
                  'scalePos', 'none');

subdir= [fileparts(file) '/'];
is= min(find(subdir=='_'));
sbj= subdir(1:is-1);
sd= subdir;
sd(find(ismember(sd,'_/')))= [];

%% load data and calculate CSP traces
[cnt,mrk,mnt]= eegfile_loadMatlab(file);
cnt= proc_selectChannels(cnt, 'not','E*');
load([save_dir 'csp_' sd], 'csp_w', 'csp_a', 'csp_band', ...
     'csp_clab', 'mrk');
cidx= chanind(cnt, csp_clab);
w= zeros(length(cnt.clab), 2);
w(cidx,:)= csp_w;
csp_w= w;
w= zeros(2, length(cnt.clab));
w(:,cidx)= csp_a;
csp_a= w;
mnt.x= 1.2*mnt.x;
mnt.y= 1.2*mnt.y;
cnt_csp= proc_linearDerivation(cnt, csp_w);
[filt_b, filt_a]= butter(5, csp_band/cnt.fs*2);
cnt_csp= proc_filt(cnt_csp, filt_b, filt_a);

pp= percentiles(cnt_csp.x(:,1), [0.05 99.95]);;
csp1_rng= 0.9*[-1 1]*max(abs(pp));
pp= percentiles(cnt_csp.x(:,2), [0.05 99.95]);;
csp2_rng= 0.9*[-1 1]*max(abs(pp));

anim= set_ifempty(anim, ...
                  'eeg_band', csp_band, ...
                  'csp1_ylim', csp1_rng, ...
                  'csp2_ylim', csp2_rng, ...
                  'video_start', mrk.pos(1)/mrk.fs-3);

[filt_b, filt_a]= butter(5, anim.eeg_band/cnt.fs*2);
cnt= proc_filt(cnt, filt_b, filt_a);
if isequal(anim.eeg_band, csp_band),
  cnt_flt= cnt;
else
  [filt_b, filt_a]= butter(5, csp_band/cnt.fs*2);
  cnt_flt= proc_filt(cnt, filt_b, filt_a);
end

if isempty(anim.eeg_scale),
  ct= proc_selectChannels(cnt, anim.eeg_clab);
  pp= percentiles(ct.x(:), [0.5 99.5]);
  anim.eeg_scale= 3*max(abs(pp));
  clear ct
end


%% generate figure of requested size
clf;
set(gcf, 'Units','Pixel', 'MenuBar','none', ...
         'Pointer','custom', 'PointerShapeCData',ones(16)*NaN, ...
         'Renderer','painters', 'DoubleBuffer','on', ...
         'Clipping','off', 'HitTest','off', 'Interruptible','off');
pos_sc= get(0, 'ScreenSize');
newpos= [5 pos_sc(4)-24-video.size(2) video.size];
set(gcf, 'Position',newpos);
drawnow;
set(gcf, 'Position',newpos);  %% hack needed due to error in fluxbox
drawnow;

%% draw CSP filters or patterns 
if anim.draw_filters,
  w= csp_w;
else
  w= csp_a;
end
h_csp1_map= subplotxl(3, 1, 1, [0.05], [0.05 0 0.75]);
h_csp1_scalp= scalpPlot(mnt, w(:,1), opt_scalp);
set([h_csp1_scalp.head h_csp1_scalp.nose], ...
    'Color', anim.csp1_outline_color, ...
    'LineWidth', anim.csp_outline_width);
h_csp2_map= subplotxl(3, 1, 3, [0.05], [0.05 0 0.75]);
h_csp2_scalp= scalpPlot(mnt, w(:,2), opt_scalp);
set([h_csp2_scalp.head h_csp2_scalp.nose], ...
    'Color', anim.csp2_outline_color, ...
    'LineWidth', anim.csp_outline_width);
h_eeg_map= subplotxl(3, 1, 2, [0.05], [0.05 0 0.75]);


%% plot whole traces of which on small windows will be visible at time
iv_scalp= floor(anim.ival_scalp(1)*cnt.fs):ceil(anim.ival_scalp(2)*cnt.fs);
T= size(cnt.x,1);
h_csp1_ax= subplotxl(3, 1, 1, [0.05], [0.3 0 0]);
line([0 T],[0 0], 'Color',0.7*[1 1 1]);
hold on;
xx= iv_scalp([1 end]);
if anim.indicate_ival,
  h_csp1_ival= line([xx; xx], anim.csp1_ylim([1 1; 2 2]), ...
                    'Color','k', 'LineStyle',':');
end
h_csp1_trace= plot(cnt_csp.x(:,1));
set(h_csp1_trace, 'Color', anim.csp1_trace_color);
set(h_csp1_ax, 'YLim', anim.csp1_ylim);
hold off;

h_csp2_ax= subplotxl(3, 1, 3, [0.05], [0.3 0 0]);
line([0 T],[0 0], 'Color',0.7*[1 1 1]);
hold on;
if anim.indicate_ival,
  h_csp2_ival= line([xx; xx], anim.csp2_ylim([1 1; 2 2]), ...
                    'Color','k', 'LineStyle',':');
end
h_csp2_trace= plot(cnt_csp.x(:,2));
set(h_csp2_trace, 'Color', anim.csp2_trace_color);
set(h_csp2_ax, 'YLim', anim.csp2_ylim);
hold off;

h_eeg_ax= subplotxl(3, 1, 2, [0.05], [0.3 0 0]);
chans= chanind(cnt, anim.eeg_clab);
nChans= length(chans);
hold on;
yLim= [-0.75 nChans-0.25]*anim.eeg_scale;
set(h_eeg_ax, 'YLim',yLim, ...
              'YDir','reverse');
for cc= 1:nChans,
  y0= (cc-1)*anim.eeg_scale;
  line([0 T],[y0 y0], 'Color',0.7*[1 1 1]);
  h_eeg_trace(cc)= plot(y0 + cnt.x(:,chans(cc)));
end

hnds= [h_csp1_ax h_csp2_ax h_eeg_ax];
if isempty(anim.window_length),
  set(hnds, 'Unit','Pixel');
  pos= get(h_csp1_ax, 'Position');
  pos([1 3])= round(pos([1 3]));
  set(h_csp1_ax, 'Position',pos);
  po= round(get(h_csp2_ax, 'Position'));
  set(h_csp2_ax, 'Position',[pos(1) po(2) pos(3) po(4)]);
  po= round(get(h_eeg_ax, 'Position'));
  set(h_eeg_ax, 'Position',[pos(1) po(2) pos(3) po(4)]);
  winlen_sa= pos(3);
else
  winlen_sa= anim.window_length*cnt.fs;
end
offset= floor(anim.video_start * video.fps);
nFrames= ceil(video.duration * video.fps);
set(hnds, 'Box','on', ...
          'XLim', offset/video.fps*cnt.fs + [1 winlen_sa], ...
          'XTick',[], ...
          'YTick',[]);




xx= iv_scalp([1 1 end end]);
h_eeg_shade= patch(xx([1 1 2 2]), yLim([1 2 2 1]), anim.shade_color(3,:));
%set(h_eeg_shade, 'EdgeColor','none');
moveObjectBack(h_eeg_shade);
hold off;


%% Show Class makers
ival= anim.video_start + [0 video.duration];
ival= ival*mrk.fs;
iVisible= find(mrk.pos>=ival(1) & mrk.pos<=ival(2));
mrk= mrk_selectEvents(mrk, iVisible, 'singleton');
csp_hnds= [h_csp1_ax h_csp2_ax];
ytext= [anim.csp1_ylim(2) anim.csp2_ylim(1)];
vali= {'bottom','top'};
ylims= [anim.csp1_ylim; anim.csp2_ylim];
h_tag= zeros(length(mrk.pos), 2);
for ii= 1:length(mrk.pos),
  cli= [1 2]*mrk.y(:,ii);
  tag= sprintf(anim.tag_names, mrk.className{cli});
  for ax= 1:2,
    axes(csp_hnds(ax));
    xx= mrk.pos(ii) + anim.shade_ival/1000*cnt.fs;
    if anim.show_class_tags,
      h_tag(ii,ax)= text(xx(1), ytext(ax), tag);
      set(h_tag(ii,ax), 'HorizontalAli','left', ...
                        'VerticalAli',vali{ax}, ...
                        anim.tag_fontspec{:});
    end
    hp= patch(xx([1 1 2 2]), ylims(ax,[1 2 2 1]), anim.shade_color(cli,:));
%    set(hp, 'EdgeColor','none');
    moveObjectBack(hp);
  end
end


%% prepare saving of the moving file
if video.save,
  movienr= 1;
  movie= avifile([video.save_dir 'csp_tmp'], ...
                 'Compression','none', 'Quality',100, 'fps',video.fps);
end
figure(gcf);  

waitForSync;
h_eeg_scalp= struct('ax',[]);
for ff= [1:nFrames] + offset,
  %% move eeg and csp traces
  x0= round((ff-1)/video.fps*cnt.fs + anim.speed_factor);
  xLim= x0 + [0 winlen_sa-1];
  set(hnds, 'XLim',xLim);

  %% place shaded area in eeg at correct position
  idx= x0 + iv_scalp;
  set(h_eeg_shade, 'XData',idx([1 1 end end]));
  
  %% update scalp map
  axes(h_eeg_map);
  pat= log(var(cnt_flt.x(idx,:)));
  delete_hnd(rmfield(h_eeg_scalp,'ax'));
  h_eeg_scalp= scalpPlot(mnt, pat, opt_scalp, ...
                         'mark_channels',anim.eeg_clab, ...
                         'colAx',anim.eeg_colrng);
  set([h_eeg_scalp.head h_eeg_scalp.nose], ...
      'Color', anim.eeg_outline_color, ...
      'LineWidth', anim.csp_outline_width);

  %% update position of interval indicator
  if anim.indicate_ival,
    set([h_csp1_ival(1) h_csp2_ival(1)], 'XData',idx([1 1]));
    set([h_csp1_ival(2) h_csp2_ival(2)], 'XData',idx([end end]));
  end
  
  %% make visible only those class tags that should be these
  tagpos= mrk.pos + anim.shade_ival(1)/1000*cnt.fs;
  iVisible= find(tagpos>=xLim(1) & tagpos<xLim(2));
  set(h_tag, 'Visible','off');
  set(h_tag(iVisible,:), 'Visible','on');
  
  %% save frame to movie file
  drawnow;
  if video.save,
    %% add current frame to the movie file
    F= getframe(gcf);
    movie= addframe(movie, F);
    %% if movie file is too large or all frames written close movie and compress
    sz= movie.Height*movie.Width*movie.TotalFrames*3;
    if sz>=video.maxFileSize | ff==nFrames+offset,
      movie= close(movie);
      movie_name= sprintf('csp_tmp_%03d.avi',movienr);
      if video.compress,
        cmd= sprintf('cd %s; transcode -i csp_tmp.avi %s -o %s',...
                     video.save_dir, video.codec_spec, movie_name);
        unix_cmd(cmd, 'using transcode');
      else
        cmd= sprintf('cd %s; mv csp_tmp.avi %s', video.save_dir, ...
                     video.save_name, movie_name);
        unix_cmd(cmd, 'renaming AVI file');
      end
      %% if not all frames were written, open another movie file
      if ff<nFrames+offset,
        movienr= movienr + 1;
        movie= avifile([video.save_dir 'csp_tmp.avi'], ...
                       'Compression','none', 'Quality',100, 'fps',video.fps);
      end
    end
  else
    waitForSync(1000/video.fps);
  end
  
end


if video.save,
  movie_name= sprintf(video.save_name, sd);
  if movienr>1,
    cmd= sprintf('cd %s; avimerge -o %s -i csp_tmp_???.avi', ...
                 video.save_dir, movie_name);
    unix_cmd(cmd, 'merging AVI files');
    cmd= sprintf('cd %s; rm -f csp_tmp_???.avi', video.save_dir);
    unix_cmd(cmd, 'deleting temp files');
  else
    cmd= sprintf('cd %s; mv csp_tmp_001.avi %s', video.save_dir, movie_name);
    unix_cmd(cmd, 'renaming video file');
  end
  cmd= sprintf('cd %s; rm -f csp_tmp.avi', video.save_dir);
%  unix_cmd(cmd, 'deleting temp files');
end
