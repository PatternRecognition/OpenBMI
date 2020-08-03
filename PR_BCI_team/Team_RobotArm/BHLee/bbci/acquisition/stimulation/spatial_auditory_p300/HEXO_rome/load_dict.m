function dict = load_dict(varargin),

opt= propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
                  'store', false, ...
                  'directory', 'temp');

% own_dir = [fileparts(which('load_dict')) '\'];              
% dict = load([own_dir 'dict.mat'], 'dict');
dict = [];
              
if opt.store,
	if ~isdir(opt.directory),
        mkdir(opt.directory);
    end
    save([opt.directory '\dict.mat'], 'dict');
end

end