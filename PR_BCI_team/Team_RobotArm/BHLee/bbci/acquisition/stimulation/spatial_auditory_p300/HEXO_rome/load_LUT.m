function LUT = create_LUT(varargin),

global TODAY_DIR;

opt= propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
                  'store', false, ...
                  'directory', TODAY_DIR);

LUT = struct();

eval(strcat('LUT_', opt.language));

if opt.store,
	if ~isdir(opt.directory),
        mkdir(opt.directory);
    end
    save([opt.directory '\LUT.mat'], 'LUT');
end

end