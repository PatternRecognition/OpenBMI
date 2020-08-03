function eb = subdir2expbase(subdir_list, varargin)
% sudbir2expbase - Separate subject and date from subdir_list
% eb = subdir2expbase(subdir_list, varargin)
%
% Ryota Tomioka, June 2007.

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'paradigm', '', 'appendix', '');

eb = repmat(struct('subject',[],'date',[], 'paradigm', opt.paradigm, 'appendix', opt.appendix), size(subdir_list));

for i=1:prod(size(subdir_list))
  subdir= subdir_list{i};
  if subdir(end)=='/'
    subdir = subdir(1:end-1);
  end
    
  is= min(find(subdir=='_'));
  eb(i).subject = subdir(1:is-1);
  eb(i).date    = subdir(is+1:end);
end
