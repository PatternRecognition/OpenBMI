N= 150;

opt= struct('perc_dev', 0.2);
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 2000;
opt.duration_cue= 350;
opt.position= VP_SCREEN;
opt.filename= 'braking';

opt.handle_background= stimutil_initFigure(opt, ...
  'background',[1 1 1]);

H= stimutil_drawPicture({'car_green_lights.png', ...
                         'car_red_lights.png'},...
                        'image_size', 0.33, ...
                        'pic_base', [BCI_DIR '/acquisition/data/'], ...
                        'pic_dir', 'images/');
opt.cue_std= line([-1 -1],[1 1], 'Color','w');
opt.cue_dev= H.image(2);
opt.handle_cross= H.image(1);

%fprintf('for testing:\n  stim_oddballVisual(N, opt, ''test'',1);\n');
%fprintf('stim_oddballVisual(N, opt);\n');
