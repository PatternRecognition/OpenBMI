function h= stimutil_readimages(filespec, varargin)

global DATA_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'folder', [DATA_DIR 'images/']);

dd= dir([opt.folder '/' filespec]);
folder= fileparts([opt.folder '/' filespec]);
set(gca, 'NextPlot','add');
for ii= 1:length(dd),
  im= imread([folder '/' dd(ii).name]);
  h(ii)= imagesc(im);
end
set(h, 'Visible','off');
