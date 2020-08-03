function fv= load_uci_dataset(dataset, varargin)
%fv= load_uci_dataset(dataset, <opts>)
%
% IN  dataset- either the name of the data set (relative to
%              [DATA_DIR 'uci/'], or
%              index of the data set. In the latter case the files
%              of interest can be specified by further properties:
%     opts   struct (or property/value list) with fields
%      .missing_features  - if 0, only data sets with complete information
%                           are enumerated
%      .symbolic_features - if 0, only data sets without symbolic features
%                           are enumerated (but note that in the other case
%                           symbolic features are converted to integers).
%
% OUT  fv    feature vector struct with fields
%       .x         - data matrix [nDim nSamples]
%       .y         - label affiliation matrix [nClasses nSamples]
%       .className - cell array of class names
%       .title     - name of the data set
%
% When argument dataset is empty, the number of data sets satisfying the
% properties defined by opts is returned.
%
% GLOBZ DATA_DIR
%
% See also load_features_from_ascii

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'missing_features', 0, ...
                  'symbolic_features', 0);


set_list= {'ecoli/ecoli.data', [9 1], 0, 0,
           'german/german.data', 25, 0, 0,
           'glass/glass.data', [11 1], 0, 0,
           'heart/heart.data', 14, 0, 0,
           'image/segmentation.data', 1, 0, 0,
           'ionosphere/ionosphere.data', 35, 0, 0,
           'iris/iris.data', 5, 0, 0,
           'letter/letter-recognition.data', 1,  0, 0,
           'optdigits/optdigits.tra', 65, 0, 0,
           'pendigits/pendigits.tra', 17, 0, 0,
           'pima-diabetes/pima-indians-diabetes.data', 9, 0, 0,
           'satelite/sat.trn', 37, 0, 0,
           'thyroid/new-thyroid.data', 1, 0, 0,
           'waveform/waveform.data', 22, 0, 0,
           'wine/wine.data', 1, 0, 0,
           'yeast/yeast.data', [10 1], 0, 0,
%%sets_with_missing
           'car/car.data', 7, 1, 1,
           'hepatitis/hepatitis.data', 1, 1, 1,
           'mushroom/agaricus-lepiota.data', 1, 1, 1,
           'soybean/soybean-large.data', 1, 1, 1};


if ischar(dataset) & ~isempty(dataset),
  ii= strmatch(dataset, set_list(:,1));
else
  ii= dataset;
  idx= 1:length(set_list);
  missing= [set_list{:,3}];
  symbolic= [set_list{:,4}];
  if ~opt.missing_features,
    idx= setdiff(idx, find(missing));
  end
  if ~opt.symbolic_features,
    idx= setdiff(idx, find(symbolic));
  end
  if isempty(dataset),
    fv= length(idx);
    return;
  end
  set_list= set_list(idx,:);
end
dataset= set_list{ii,1};

global DATA_DIR
file= [DATA_DIR 'uci/' dataset];

opt.label_idx= set_list{ii,2}(1);
opt.ignore_idx= set_list{ii,2}(2:end);

fv= load_features_from_ascii(file, opt);
aa= max(find(dataset=='/'));
zz= find(dataset=='.');
fv.title= dataset(aa+1:zz-1);
