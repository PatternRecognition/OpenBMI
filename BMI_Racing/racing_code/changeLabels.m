function out=changeLabels(data,varargin)

% e.g. out=changeLabels(data,{'right',1;'others',2});
% 일단은 cls/others 경우만

opt=opt_cellToStruct(varargin{:});
cls_def=fieldnames(opt);
i=find(ismember(cls_def,data.class));
j=find(~ismember(cls_def,data.class));
class_name=cls_def{i};
rest_name=cls_def{j};
class_num=opt.(char(cls_def(ismember(cls_def,data.class))));
rest_num=opt.(char(cls_def(~ismember(cls_def,data.class))));

orig_class_num=find(ismember(data.class,class_name))-size(data.class,1);
% 원래 클래스 라벨이 1,2,3,4... 이런 식으로 순서대로 되어 있어야 함
orig_rest_num=1:size(data.class,1);
orig_rest_num(orig_rest_num==orig_class_num)=[];

out=rmfield(data,{'y_dec','y_logic','y_class','class'});

idx_cls=find(data.y_dec==orig_class_num);
idx_rest=find(ismember(data.y_dec,orig_rest_num));

out.y_dec=rest_num*ones(size(data.y_dec));
out.y_dec(idx_cls)=class_num;

out.y_logic=zeros(max(class_num,rest_num),size(data.y_logic,2));
logic1=data.y_logic(orig_class_num,:);
logic2=logical(sum(data.y_logic(orig_rest_num,:)));
out.y_logic(class_num,:)=logic1;
out.y_logic(rest_num,:)=logic2;

% 문자 타입이 이상
out.y_class=cell(1,length(data.y_class));
out.y_class(idx_cls)={cls_def(i)};
out.y_class(idx_rest)={cls_def(j)};

% 문자 타입이 이상
out.class={opt.(cls_def{i}),cls_def{i};opt.(cls_def{j}),cls_def{j}};
