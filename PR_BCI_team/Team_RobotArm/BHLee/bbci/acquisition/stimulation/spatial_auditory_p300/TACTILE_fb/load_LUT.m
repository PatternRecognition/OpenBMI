function LUT = create_LUT(varargin),

opt= propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
                  'store', false, ...
                  'directory', 'temp');

LUT = struct();

state = 1;
LUT(state).direction(1).label = '';
LUT(state).direction(1).nState = 1;
LUT(state).direction(1).type = 'navi';
LUT(state).direction(2).label = '';
LUT(state).direction(2).nState = 1;
LUT(state).direction(2).type = 'navi';
LUT(state).direction(3).label = '';
LUT(state).direction(3).nState = 1;
LUT(state).direction(3).type = 'navi';
LUT(state).direction(4).label = '';
LUT(state).direction(4).nState = 1;
LUT(state).direction(4).type = 'navi';
LUT(state).direction(5).label = '';
LUT(state).direction(5).nState = 1;
LUT(state).direction(5).type = 'navi';
LUT(state).direction(6).label = '';
LUT(state).direction(6).nState = 1;
LUT(state).direction(6).type = 'navi';
LUT(state).direction(7).label = '';
LUT(state).direction(7).nState = 1;
LUT(state).direction(7).type = 'navi';
LUT(state).direction(8).label = '';
LUT(state).direction(8).nState = 1;
LUT(state).direction(8).type = 'navi';

if opt.store,
	if ~isdir(opt.directory),
        mkdir(opt.directory);
    end
    save([opt.directory '\LUT.mat'], 'LUT');
end

end