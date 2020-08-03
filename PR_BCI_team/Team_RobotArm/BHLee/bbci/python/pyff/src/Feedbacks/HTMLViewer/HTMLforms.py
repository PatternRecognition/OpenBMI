import wx
import wx.html

FormSubmitEventType = wx.NewEventType()

EVT_FORM_SUBMIT = wx.PyEventBinder(FormSubmitEventType)

# Global variable set to 1 if the current HTML file contains a form
ISFORM = 2

__SENTINEL = object()

def GetParam(tag, param, default=__SENTINEL):
    """ Convenience function for accessing tag parameters"""
    if tag.HasParam(param):
        return tag.GetParam(param)
    else:
        if default == __SENTINEL:
            raise KeyError
        else:
            return default


class FormSubmitEvent(wx.PyEvent):
    """
        Event indication a form was submitted.
        form is the form object being submitted
        args is a dict of form arguments
    """
    def __init__(self, form, args):
        wx.PyEvent.__init__(self)
        self.SetEventType(FormSubmitEventType)
        self.form = form
        self.args = args
    
class HTMLForm(object):
    def __init__(self, tag, container):
        self.container = container
        self.fields = []
        self.action = GetParam(tag, "ACTION", default=None)
        self.method = GetParam(tag, "METHOD", "GET")
        if self.method not in ("GET", "POST"):
            self.method = "GET"
            
    def hitSubmitButton(self):
        for field in self.fields:
            if isinstance(field, SubmitButton):
                field.OnClick(None)
                return
                
    def submit(self, btn=None):
        args = self.createArguments()
        if btn and btn.name:
            args[btn.name] = btn.GetLabel()
        evt = FormSubmitEvent(self, args)
        self.container.ProcessEvent(evt)
        
    def createArguments(self):
        args = {}
        for field in self.fields:
            if field.name and field.IsEnabled():
                val = field.GetField()
                if val is None:
                    continue
                args[field.name] = val
        return args



class FormTagHandler(wx.html.HtmlWinTagHandler):
    typeregister = {}
    
    
    @classmethod
    def registerType(klass, ttype, controlClass):
        klass.typeregister[ttype] = controlClass
    
    def __init__(self):
        self.form = None
        wx.html.HtmlWinTagHandler.__init__(self)
    
    def GetSupportedTags(self):
        return "FORM,INPUT,TEXTAREA,SELECT,OPTION"
        
    def HandleTag(self, tag):
        try:
            handler = getattr(self, "Handle"+tag.GetName().upper())
            return handler(tag)
        except:
            import traceback
            traceback.print_exc()
        
    def HandleFORM(self, tag):
        global ISFORM
        ISFORM = 1
        self.form = HTMLForm(tag, self.GetParser().GetWindowInterface().GetHTMLWindow())
        self.cell = self.GetParser().OpenContainer()
        self.ParseInner(tag)
        self.GetParser().CloseContainer()
        self.form = None
        self.optionList = []
        return True
    
    def HandleINPUT(self, tag):
        if tag.HasParam("type"):
            ttype = tag.GetParam("type").upper()
        else:
            ttype = "TEXT"
        klass = self.typeregister[str(ttype)]
        self.createControl(klass, tag)
        return False
        
    def HandleTEXTAREA(self, tag):
        klass = self.typeregister["TEXTAREA"]
        self.createControl(klass, tag)
        #Don't actually call ParseInner, but lie and said we did.
        #This should skip us ahead to the next tag, and let us 
        #retrieve the text verbatem from the text area
        return True
    
    def HandleSELECT(self, tag):

        if tag.HasParam("MULTIPLE"):
            pass
        self.optionList = []
        #gather any/all nested options
        self.ParseInner(tag)
        parent = self.GetParser().GetWindowInterface().GetHTMLWindow()
        if 'wxMSW' in wx.PlatformInfo:
            #HAX: wxChoice has some bizarre SetSize semantics that
            #interact poorly with HtmlWidgetCell. Most notably, resizing the control
            #triggers a paint event (in the parent, I guess?) which in turn calls Layout()
            #which calls SetSize again and so on. An extra "layer" between the control
            #and the window keeps the recursion from happening.
            myobject = wx.Panel(parent)
            selector = SingleSelectControl(myobject, self.form, tag, self.GetParser(), self.optionList)
            sz = wx.BoxSizer()
            sz.Add(selector, 1, wx.EXPAND)
            myobject.SetSizer(sz)
            myobject.SetSize(selector.GetSize())
        else:
            myobject = SingleSelectControl(parent, self.form, tag, self.GetParser(), self.optionList)
        cell = self.GetParser().GetContainer()
        
        
        cell.InsertCell(
            wx.html.HtmlWidgetCell(myobject)
        )
        self.optionList = []
        return True
        
    def HandleOPTION(self, tag):
        self.optionList.append(tag)
        return True
        
    def createControl(self, klass, tag):
        parent = self.GetParser().GetWindowInterface().GetHTMLWindow()
        myobject = klass(parent, self.form, tag, self.GetParser())
        if not isinstance(myobject, wx.Window):
            return
        cell = self.GetParser().GetContainer()
        cell.InsertCell(
            wx.html.HtmlWidgetCell(myobject)
        )
        
        
        
        
wx.html.HtmlWinParser_AddTagHandler(FormTagHandler)


## ******* HTML INPUT **********


def TypeHandler(typeName):
    """ A metaclass generator. Returns a metaclass which
    will register it's class as the class that handles input type=typeName
    """
    def metaclass(name, bases, ddict):
        klass = type(name, bases, ddict)
        FormTagHandler.registerType(typeName.upper(), klass)
        return klass
    return metaclass

class FormControlMixin(object):
    """ Mixin provides some stock behaviors for
    form controls:
        Add self to the form fields
        Setting the name attribute to the name parameter in the tag
        Disabled attribute
        OnEnter and OnClick methods for binding by 
        the actual control
    """
    def __init__(self, form, tag):
        if not form:
            return
        self.__form = form
        self.name = GetParam(tag, "NAME", None)
        form.fields.append(self)
        if tag.HasParam("DISABLED"):
            wx.CallAfter(self.Disable)
    def OnEnter(self, evt):
        self.__form.hitSubmitButton()
    def OnClick(self, evt):
        self.__form.submit(self)

class SubmitButton(wx.Button, FormControlMixin):
    __metaclass__ = TypeHandler("SUBMIT")
    def __init__(self, parent, form, tag, parser, *args, **kwargs):
        label = GetParam(tag, "VALUE", default="Submit Query")
        kwargs["label"] = label
        wx.Button.__init__(self, parent, *args, **kwargs)
        FormControlMixin.__init__(self, form, tag)
        self.SetSize((int(GetParam(tag, "SIZE", default=-1)), -1))
        self.Bind(wx.EVT_BUTTON, self.OnClick)
    def GetField(self):
        return None
        

class TextInput(wx.TextCtrl, FormControlMixin):
    __metaclass__ = TypeHandler("TEXT")
    def __init__(self, parent, form, tag, parser, *args, **kwargs):
        style = kwargs.get("style", 0)
        if tag.HasParam("READONLY"):
                style |= wx.TE_READONLY
        if form:
            style |= wx.TE_PROCESS_ENTER
        kwargs["style"] = style
        wx.TextCtrl.__init__(self, parent, *args, **kwargs)
        FormControlMixin.__init__(self, form, tag)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)
        self.SetValue(GetParam(tag, "VALUE", ''))
        ml = int(GetParam(tag, "MAXLENGTH", 0))
        self.SetMaxLength(ml)
        if ml and len(self.GetValue()) > ml:
            self.SetValue(self.GetValue()[:ml])
        if tag.HasParam("SIZE"):
            size = max(int(tag.GetParam("SIZE")), 5)
            width = self.GetCharWidth() * size
            self.SetSize((width, -1))
    def GetField(self):
    	return self.GetValue()
            
            
            
class PasswordInput(TextInput):
    __metaclass__ = TypeHandler("PASSWORD")
    def __init__(self, parent, form, tag, parser):
        TextInput.__init__(self, parent, form, tag, parser, style=wx.TE_PASSWORD)
    def GetField(self):
    	return None       
       
class Checkbox(wx.CheckBox, FormControlMixin):
    __metaclass__ = TypeHandler("CHECKBOX")
    def __init__(self, parent, form, tag, parser, *args, **kwargs):
        wx.CheckBox.__init__(self, parent, *args, **kwargs)
        FormControlMixin.__init__(self, form, tag)
        self.value = GetParam(tag, "VALUE", "1")
        if tag.HasParam("checked"):
            self.SetValue(True)
    def GetField(self):
        if self.IsChecked():
            return self.value
        else:
            return None


RadioButtonGroups=['']            
class RadioButtons(wx.RadioButton, FormControlMixin):
    __metaclass__ = TypeHandler("RADIO")
    def __init__(self, parent, form, tag, parser, *args, **kwargs): 
    	style = kwargs.get("style", 0)   
    	RadioButtonGroups.append(GetParam(tag, "NAME", ' ')	)
    	if not RadioButtonGroups[-2]== GetParam(tag,"NAME",''):
    		style |= wx.RB_GROUP
    	kwargs["style"] = style
    	wx.RadioButton.__init__(self, parent, *args, **kwargs)
        FormControlMixin.__init__(self, form, tag)
        self.value = GetParam(tag, "VALUE", " ")
    	if tag.HasParam("checked"):
    		self.setValue(True)

    def GetField(self):
    	if self.GetValue():
        	return self.value
        else:
            return None



class HiddenControl(wx.EvtHandler, FormControlMixin):
    __metaclass__ = TypeHandler("HIDDEN")
    def __init__(self, parent, form, tag, parser, *args, **kwargs):
        wx.EvtHandler.__init__(self)
        FormControlMixin.__init__(self, form, tag)
        self.value = GetParam(tag, "VALUE", "")
        self.enabled = True
    def GetField(self):
        return self.value
    def Disable(self):
        self.enabled = False
    def IsEnabled(self):
        return self.enabled
        
class TextAreaInput(wx.TextCtrl, FormControlMixin):
    __metaclass__ = TypeHandler("TEXTAREA")
    def __init__(self, parent, form, tag, parser, *args, **kwargs):
        style = wx.TE_MULTILINE
        if tag.HasParam("READONLY"):
            style |= wx.TE_READONLY
        wx.TextCtrl.__init__(self, parent, style=style)
        FormControlMixin.__init__(self, form, tag)
        if tag.HasEnding():
            src = parser.GetSource()[tag.GetBeginPos():tag.GetEndPos1()]
        else:
            src = ''
        self.SetFont(wx.SystemSettings.GetFont(wx.SYS_ANSI_FIXED_FONT))
        self.SetValue(src)
        cols = GetParam(tag, "COLS", 22)
        width = self.GetCharWidth() * int(cols)
        rows = GetParam(tag, "ROWS", 3)
        height = self.GetCharHeight() * int(rows)
        self.SetSize((width, height))

    def GetField(self):
    	return self.GetValue()


## ********* HTML SELECT ***********
class SingleSelectControl(wx.Choice, FormControlMixin):
    def __init__(self, parent, form, tag, parser, optionList, **kwargs):
        FormControlMixin.__init__(self, form, tag)
        self.values = []
        contents = []
        selection = 0
        for idx, option in enumerate(optionList):
            contents.append(parser.GetSource()[option.GetBeginPos():option.GetEndPos1()])
            self.values.append(GetParam(option, 'VALUE', ''))
            if option.HasParam("SELECTED") and not selection:
                selection = idx
        wx.Choice.__init__(self, parent, 
            choices = contents,
        )
        self.SetSelection(selection)
        
    def GetField(self):
        sel = self.GetSelection()
        value = self.values[sel]
        return value
        
        
 
