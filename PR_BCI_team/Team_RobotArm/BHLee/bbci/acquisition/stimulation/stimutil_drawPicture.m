function [H, h_fix]= stimutil_drawPicture(pic_list, varargin)

global DATA_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'pic_base',[DATA_DIR 'images/'],...
                  'pic_dir','stimuli',...
									'image_size', [.5], ...
									'image_pos', [.5 .5], ...
                  'image_height_factor', 1);

pic_dir= [opt.pic_base '/' opt.pic_dir];

if ischar(pic_list),
  pic_list= {pic_list};
end

for li= 1:length(pic_list),
  pic{li}= imread([pic_dir '/' pic_list{li}]);
end
sz= size(pic{1});

fp= get(gcf, 'Position');
if length(opt.image_size)==1,
  opt.image_size(2)= opt.image_size(1)/sz(2)*sz(1)/fp(4)*fp(3);
  opt.image_size(2)= opt.image_size(2)*opt.image_height_factor;
end
image_pos= [opt.image_pos-0.5*opt.image_size opt.image_size];

H.ax= axes('Position',image_pos);
set(H.ax, 'YDir','reverse', 'Visible','off');
hold on;
for li= 1:length(pic_list),
  H.image(li)= image(pic{li});
end
axis tight
set([H.image], 'Visible','off');

