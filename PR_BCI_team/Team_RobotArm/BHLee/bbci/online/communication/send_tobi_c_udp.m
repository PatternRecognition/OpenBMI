function varargout = send_tobi_c_udp(command, varargin)
%SEND_XML_UDP - Send Signal in XML Format via UDP
%
%Synopsis:
% send_xml_udp('init', hostname, port) % initialize standard classifier
% send_xml_udp('init', hostname, port, classifier_definition)
% send_xml_udp('add', VALUE1, VALUE2, ...)
% send_xml_udp('add', CLASSIFIER1, CLASS1, VALUE1, ...)
% send_xml_udp('send', VALUE1, VALUE2, ...)
% send_xml_udp('send', CLASSIFIER1, CLASS1, VALUE1, ...)
% send_xml_udp('close')
%
%Arguments:
%  hostname: String, IP-address or hostname
%  port: Integer value, number of the port
%  classifier_definition: Struct containing classifier
%     definitions. Such a definition looks as follows (x = definition):
%     x.name = String; % 'bbci_mi'
%     x.desc = String; % 'BBCI Motor Imagery classifier'
%     x.vtype = String; % one of {'Undef', 'Prob', 'Dist', 'CLbl', 'RCoe'},
%           defining the type of value the classifier produces
%     x.ltype = String; % on of {'Undef','Biosig','Custom','Class'},
%           defining the class label type
%     x.classes = Cell/String; % {'left', 'right'}
%  VALUE: Value to be set in the classifier. If multiple values are given,
%    without explicitly specifying the classifier and class name, they will
%    be assigned in order to the classifiers/classes.
%  CLASSIFIER: Name of the classifier to assign the score to.
%  CLASS: Name of the class to assign the score to.
%
% Note: the 'send' command will add the values and then send, whereas the
% 'add' command will only add the value but not send.
%
%Returns:
%  nothing
%
%Example:
% send_xml_udp('init', bbci.fb_machine, bbci.fb_port);
% send_xml_udp('init', bbci.fb_machine, bbci.fb_port, ...
%   {{'bbci_mi', 'BBCI Motor Imagery Classifier', ...
%     'dist', 'biosig', {'0x300', '0x301'}}});
% send_xml_udp({})

% Martijn 2011

global BCI_DIR
persistent socke message classifier serializer

switch command,
    case 'init',
        hostname= varargin{1};
        port= varargin{2};
        path(path, [BCI_DIR 'import/tcp_udp_ip']);
        path(path, [BCI_DIR 'online/communication/tobiC/']);
        socke= pnet('udpsocket', 1111);  %% what is this port number?
        if socke==-1,
            error('udp communication failed');
        end
        pnet(socke, 'udpconnect', hostname, port);
        message = icmessage_new();
        serializer = icserializerrapid_new(message);

        if nargin > 3 
            if isstruct(varargin{3}),
                classifier = varargin{3};
            elseif iscell(varargin{3}),
                classifier = conv_class_to_struct(varargin{3});
            else
                error('Classifier definition not formatted correctly');
            end
        else
            std_class = struct('name', 'std_classy', ...
                'desc', 'BBCI Standard Classifier', ...
                'vtype', 'dist', ...
                'ltype', 'custom', ...
                'classes', 'std_classes');            
            classifier = std_class;
        end
        for cl_id = 1:length(classifier),
            icmessage_addclassifier(message, ...
                classifier(cl_id).name, classifier(cl_id).desc, ...
                icmessage_getvaluetype(classifier(cl_id).vtype), ...
                icmessage_getlabeltype(classifier(cl_id).ltype));
            if ~iscell(classifier(cl_id).classes), classifier(cl_id).classes = {classifier(cl_id).classes}; end;
            for class_id = 1:length(classifier(cl_id).classes),
                icmessage_addclass(message, classifier(cl_id).name, classifier(cl_id).classes{class_id}, 0);
            end
            icmessage_dumpmessage(message);
        end
        
    case 'close',
        pnet(socke, 'close');
        socke= [];
        icmessage_delete(message);
        message = [];
        classifier = [];
        
    case {'add', 'send'},
        if isempty(message), 
            error('Classifier not initialized');
        elseif ischar(varargin{1}),
            for valI = 1:3:length(varargin),
                cls_id = strmatch(varargin{valI}, {classifier(:).name}, 'exact');
                if ~isempty(cls_id),
                    cl_id = strmatch(varargin{valI+1}, classifier(cls_id).classes, 'exact');
                    if ~isempty(cl_id),
                        icmessage_setvalue(message, varargin{valI}, varargin{valI+1}, varargin{valI+2});
                    else
                        warning('Classifier "%s" contains no class "%s". Value not set.', varargin{valI}, varargin{valI+1});
                    end
                else
                    warning('Classifier "%s" not initialized. Value not set.', varargin{valI});
                end
            end
        elseif isnumeric([varargin{:}]), 
            values = [varargin{:}];
            clsI = 1;
            clI = 1;
            for valI = 1:length(values),
                if ~isnan(values(valI)),
                    icmessage_setvalue(message, classifier(clsI).name, classifier(clsI).classes{clI}, values(valI));
                end
                if clI == length(classifier(clsI).classes), 
                    clsI = clsI + 1;
                    clI = 1;
                else
                    clI = clI + 1;
                end
                if valI < length(values) && clsI > length(classifier), 
                    warning('Too many values given. Can not assign more.');
                    break;
                end
            end
        else
            error('Don''t know how to process these inputs');
        end     
        if isequal(command, 'send'),
            xml_cmd = icmessage_serialize(serializer);
            pnet(socke, 'write', xml_cmd);
            pnet(socke, 'writepacket');
%             send_tobi_c_udp('reset_classifiers');
        end
        
    case 'get_handle',
        if ~isempty(message),
            varargout{1} = message;
        else
            error('Classifier not initialized yet. Nothing to return');
        end
        
    case 'reset_classifiers',
        % icmessage_reset();
        return;
        
    otherwise,
        error('No valid action given');
end
end

function classifier = conv_class_to_struct(cell_class),
    tags = {'name','desc','vtype','ltype','classes'};
    for cli = 1:length(cell_class),
        for tagi = 1:length(cell_class{cli})
            classifier(cli).(tags{tagi}) = cell_class{cli}{tagi};
        end
    end    
end
