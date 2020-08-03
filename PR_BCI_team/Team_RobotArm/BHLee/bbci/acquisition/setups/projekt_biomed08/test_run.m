VP_CODE= 'Temp';

setup_biomed08_BCI02

opt.bv_host= '';
opt.test= 1;
opt.visual_targetPresentation= 1;
opt.visual_cuePresentation= 1;     %% should be 0
stim_tactileP300(12, 5, opt)
