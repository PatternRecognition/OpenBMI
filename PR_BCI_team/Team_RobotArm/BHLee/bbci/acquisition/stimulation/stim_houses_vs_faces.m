function stim_houses_vs_faces(N, varargin);
%STIM_HOUSES_VS_FACES - Visual Stimulation with two Classes of Images 
%
%Description:
% A randomized sequence of images of two classes (faces and houses)
% are displayed, interleaved with a masking image. One image of the
% target_class is displayed at the very beginning. The task of the
% subject is to respond by keypress to the target image in the random
% sequence. The random sequence is
%
%Synopsis:
% stim_houses_vs_faces(N, OPT)
% stim_houses_vs_faces(N, TARGET_CLASS, OPT)
%
%Arguments:
% N: 1x2 vector: [Total number of stimuli, Number of target stimuli],
%    If N is a scalar, the number of target stimuli is set to 10% of N.
% TARGET_CLASS: 1 or 2 indicating the target class to which the subject
% OPT: Struct or property/value list of optional properties:
%  'target_class': 1 or 2; 0 means no target image is shown, default 0
%  'isi': inter-stimulus interval [ms], default 2000.
%  'duration_cue': duration for which the stimulus is shown [ms], default 500.
%  'duration_target': duration for which the target image is show in the
%     beginning [ms], default: 3000.
%  'perc_class1': percentage of images of class 1, default .5
%  'image_pos': relative position of the center of the image on the screen,
%     default [0.5 0.5].
%  'image_size': relative size of the image. If this is a scalar, it is
%     taken as relative width, while the relative height is calculated
%     according to the format of the masking image, default 0.3.
%  'pic_base': folder in which data bases of images are stored, 
%     default [DATA_DIR 'images'] (DATA_DIR is a global variable).
%  'pic_dir': subfolder which searched for images, default 'houses_vs_faces'.
%  'class1': pattern by which image files of class1 are defined, 
%     default 'face*'.
%  'class2': pattern by which image files of class2 are defined,
%     default 'house*'.
%  'mask_image': file name of the masking image, default 'mask.bmp'.
%  'fixation': switch to show (1) or not to show (0) a fixation cross.
%  'fixation_vpos': relative vertical position on the screen of the
%     fixation cross, default 0.57.
%  'fixation_size': size of fixation cross (horizontal direction, relative
%     to screen; vertical size is matched), default 0.04.
%  'fixation_spec': specification of line properties of the fixation cross,
%     default: {'LineWidth',12, 'Color',[.9 0 0]}.
%  'msg_vpos': relative vertical position on the screen of messages,
%     default: 0.57.
%  'msg_spec': specification of text properties of messages,
%     default: {'FontSize',0.1, 'FontWeight','bold', 'Color',[.9 0 0]}.
%
%Comment: This function was written ad-hoc. Maybe it can be rewritten
%  more elegantly, such that it uses stimutil_drawPicture and stim_visualCue.
%
%Markers sent to parallel port:
%  1: nontarget image of class 1
%  2: nontarget image of class 2
% 11: target image of class 1
% 12: target image of class 2
% 21: initial presentation of target image of class 1
% 22: initial presentation of target image of class 2
% 60: image off
%101-126: response key (101='a', ..., 126='z')
%251: beginning of initial relax period
%252: start of stimulus sequence (after initial target presentation)
%253: beginning of final relax period
%254: end of final relax period

% blanker@cs.tu-berlin.de

global DATA_DIR

if nargin==0 | isempty(N),
  N= [60 5];
end

if length(varargin)>0 & isnumeric(varargin{1}),
  opt= propertylist2struct('target_class',varargin{1}, varargin{2:end});
else
  opt= propertylist2struct(varargin{:});
end

opt= set_defaults(opt, ...
									'target_class', 0, ...
                  'breakFactor', 1,...
                  'perc_class1', 0.5, ...
                  'pic_base',[DATA_DIR 'images/'],...
                  'pic_dir','houses_vs_faces',...
									'class1', 'face*', ...
									'class2', 'house*', ...
                  'mask_image', 'mask.bmp', ...
                  'duration_cue', 500,...
									'duration_target', 7000, ...
									'duration_msg_target', 1000, ...
									'duration_msg_go', 1000, ...
                  'isi', 2000,...
									'image_size', [.3], ...
									'image_pos', [.5 .5], ...
                  'image_height_factor', 1, ...
									'color_background',[0 0 0], ...
                  'countdown', 3, ...
									'msg_relax', 'entspannen', ...
									'msg_target', 'Zielbild', ...
									'msg_go', 'los geht''s', ...
                  'fixation', 1, ...
                  'fixation_vpos', 0.57, ...
                  'fixation_size', 0.04, ...
                  'fixation_spec', {'LineWidth',12, 'Color',[.9 0 0]}, ...
									'msg_vpos', 0.57, ...
									'msg_spec', {'FontSize',0.1, 'FontWeight','bold', ...
                                'Color',[.9 0 0]});

if length(N)==1,
  N= [N ceil(N*0.1)];
end
if opt.target_class==0,
  N(2)= 0;
  %% for the time being:
  error('opt.target_class must be non-zero');
end

dur= N(1)*opt.isi/1000 + 2 + 15*opt.breakFactor;
if opt.target_class>0,
  dur= dur + ...
       (opt.duration_target+opt.duration_msg_target+opt.duration_msg_go)/1000;
end
fprintf('approximate duration: %.1fs\n', dur);

pic_dir= [opt.pic_base '/' opt.pic_dir];
dd= dir([pic_dir '/' opt.class1]);
list1= {dd.name};
dd= dir([pic_dir '/' opt.class2]);
list2= {dd.name};
list= cat(2, list1, list2);
L1= length(list1);
L2= length(list2);
im= imread([pic_dir '/' opt.mask_image]);

if opt.target_class>0,
	if opt.target_class==1,
		target_no= ceil(rand*L1);
	else
		target_no= L1 + ceil(rand*L2);
	end
  pic_name= list{target_no};
end

%% choose sequence of stimuli
N_cl(1)= round(N(1)*opt.perc_class1);
N_cl(2)= N(1)-N_cl(1);
N_cl(opt.target_class)= N_cl(opt.target_class) - N(2);
L1a= L1-(opt.target_class==1);
seq1= rand_seq(N_cl(1), L1a);
seq2= L1a + rand_seq(N_cl(2), L2-(opt.target_class==2));
seq= rand_combine_seq(seq1, seq2);
if opt.target_class>0,
  idx= find(seq>=target_no);
  seq(idx)= seq(idx) + 1;
  seq= rand_combine_seq(seq, target_no*ones(1,N(2)));
end

%% prepare graphics including mask image
clf;
ppTrigger(251);
figureMaximize;
set(gcf,'Menubar','none', 'Toolbar','none', ...
        'Color',opt.color_background, 'DoubleBuffer','on', ...
        'Colormap',gray(256));
set(gcf,'KeyPressFcn','ch= get(gcf,''CurrentCharacter''); ppTrigger(ch-''a''+101);');
msg_spec= {'HorizontalAli','center', 'VerticalAli','middle', ...
            'FontUnits','normalized', ...
            opt.msg_spec{:}};
fp= get(gcf, 'Position');
if length(opt.image_size)==1,
  opt.image_size(2)= opt.image_size(1)/size(im,2)*size(im,1)/fp(4)*fp(3);
  opt.image_size(2)= opt.image_size(2)*opt.image_height_factor;
end
image_pos= [opt.image_pos-0.5*opt.image_size opt.image_size];
h_ax_mask= axes('Position',image_pos);
h_im_mask= image(im);
h_ax_im= axes('Position',image_pos);
h_ax_msg= axes('Position',[0 0 1 1]);
set([h_ax_msg h_ax_im h_ax_mask], 'Visible','off');
axes(h_ax_msg);
h_msg= text(0.5, opt.msg_vpos, opt.msg_relax, msg_spec{:});
fix_w= opt.fixation_size;
fix_h= fix_w/fp(4)*fp(3);
set(h_ax_msg, 'XLim',[0 1], 'YLim',[0 1]);
h_fix= line(0.5 + [-fix_w fix_w; 0 0]', ...
            opt.fixation_vpos + [0 0; -fix_h fix_h]', ...
            opt.fixation_spec{:});
set(h_fix, 'Visible','off');
pause(1+9*opt.breakFactor);

if opt.target_class>0,
	waitForSync;
  set(h_msg, 'String',opt.msg_target, 'Visible','on');
	drawnow;
	waitForSync(opt.duration_msg_target);
  set(h_msg, 'Visible','off');
  set(h_fix, 'Visible','on');
  pic= imread([pic_dir '/' pic_name]);
  axes(h_ax_im);
  h= image(pic);
  set(h_ax_im, 'Visible','off');
  set(h_fix, 'Visible','on');
  drawnow;
  ppTrigger(20+opt.target_class);
	waitForSync(opt.duration_target);
	delete(h);
	ppTrigger(60);
  set(h_fix, 'Visible','off');
  set(h_msg, 'String',opt.msg_go, 'Visible','on');
	drawnow;
	waitForSync(opt.duration_msg_go);
end

%stimutil_countdown(h_msg, opt.countdown);
set(h_msg, 'Visible','off');
set(h_fix, 'Visible','on');
ppTrigger(252);
waitForSync(opt.isi - opt.duration_cue);

axes(h_ax_im);
pic_name= list{seq(1)};
pic= imread([pic_dir '/' pic_name]);
waitForSync;
for ii= 1:N(1),
  class_idx= seq(ii)>L1;
  h= image(pic);
  set(h_ax_im, 'Visible','off');
  drawnow;
  ppTrigger(class_idx + 1 + 10*(seq(ii)==target_no)),
	if ii<N(1),
		pic_name= list{seq(ii+1)};
    pic= imread([pic_dir '/' pic_name]);
  end
  waitForSync(opt.duration_cue);
  delete(h);
  drawnow;
  ppTrigger(60);
  waitForSync(opt.isi - opt.duration_cue);
end

set(h_fix, 'Visible','off');
set(h_msg, 'String',opt.msg_relax, 'Visible','on');
ppTrigger(253);
pause(1+5*opt.breakFactor);
set(h_msg, 'String','fin');
delete(h_ax_mask);

ppTrigger(254);
