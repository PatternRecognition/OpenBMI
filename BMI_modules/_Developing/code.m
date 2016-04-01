개발시 숙지 할 점 (상황에 따라 변동성있게 적용할 것)


function은 EPOCHING 된 EPO 데이터를 기준으로 만들 것->segmentation 이전은 예외 
변수명 - EPO
필드 - x, t, fs, y, y_logic, chSet, class

필드는 기본적으로 위와 같이 구성되어 있으므로, 펑션을 만들 경우 고려해야할 점으로 
1. 필수적인 필드가 있는지 -> 없으면 error or break
2. 필수적이지는 않으나 중요한 필드가 빠져 있다면 -> warning

input parameter넣을 경우, 
1. 중요도 순서로 넣을 것
2. 데이터는 앞으로, 파리미터는 {} 셀 형식으로 뒤로 뺄 것 
예) 
보낼때->EPO=prep_segmentation(EEG.data, EEG.marker, {'interval', [750 3500]});
데이터인 EEG.data와 EEG.marker는 앞으로 이외의 파라미터는 {}안에 pair를 이루도록 넣을것, {'interval', [750 3500]; 'others', 100}

받을때->[ epo ] = prep_segmentation( dat, marker, varargin )
데이터 항목은 그대로 받고, 나머지 파라미터 항복은 모두 varargin으로 받을 것. 
vararing 항목은 "opt=opt_cellToStruct(varargin{:})" 을 사용하여 구조체 형식으로 받고
1. 필수파라미터를 검사할것
    예) if isfield(opt,'interval')
        ival=opt.interval
        else
        error('Parameter is missing: "interval"');
    end

        
들어오는 input 데이터의 struct를 나가는 output paremter가 모두 가지고 있어야 함
