���߽� ���� �� �� (��Ȳ�� ���� �������ְ� ������ ��)


function�� EPOCHING �� EPO �����͸� �������� ���� ��->segmentation ������ ���� 
������ - EPO
�ʵ� - x, t, fs, y, y_logic, chSet, class

�ʵ�� �⺻������ ���� ���� �����Ǿ� �����Ƿ�, ����� ���� ��� ����ؾ��� ������ 
1. �ʼ����� �ʵ尡 �ִ��� -> ������ warning or break
2. �ʼ��������� ������ �߿��� �ʵ尡 ���� �ִٸ� -> warning

input parameter���� ���, 
1. �߿䵵 ������ ���� ��
2. �����ʹ� ������, �ĸ����ʹ� {} �� �������� �ڷ� �� �� 
��) 
������->EPO=prep_segmentation(EEG.data, EEG.marker, {'interval', [750 3500]});
�������� EEG.data�� EEG.marker�� ������ �̿��� �Ķ���ʹ� {}�ȿ� pair�� �̷絵�� ������, {'interval', [750 3500]; 'others', 100}

������->[ epo ] = prep_segmentation( dat, varargin )
������ �׸��� �״�� �ް�, ������ �Ķ���� �׺��� ��� varargin���� ���� ��. 
vararing �׸��� "opt=opt_cellToStruct(varargin{:})" �� ����Ͽ� ����ü �������� �ް�
1. �ʼ��Ķ���͸� �˻��Ұ�
    ��) if isfield(opt,'interval')
        ival=opt.interval
        else
        error('Parameter is missing: "interval"');
    end

        
������ input �������� struct�� ������ output paremter�� ��� ������ �־�� ��


��������
EEG.data, EEG.marker, EEG.info  -> ���� EEG DATA
CNT -> continuous eeg data
SMT -> segmented eeg data
FT -> feature

- dat, in 
- out