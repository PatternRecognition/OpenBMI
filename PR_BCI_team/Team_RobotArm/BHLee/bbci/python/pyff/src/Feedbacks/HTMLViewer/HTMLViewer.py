# HtmlViewer.py -
# Copyright (C) 2011-2012  Matthias Treder
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301 USA.

""" 
Viewer for HTML pages. It can be used in order to display 
instructions or present questionnaires.

Requires the wxPython package which supports a subset of HTML commands: 
http://docs.wxwidgets.org/trunk/overview_html.html

Run the feedback with the sample htm pages for more information.
"""

from FeedbackBase.MainloopFeedback import MainloopFeedback
import os,wx
import wx.html

import HTMLforms

class HTMLViewer(MainloopFeedback):

    def init(self):
        ## Files and directories
        self.pages = ["ExampleInstructions.htm","SubjectInfo.htm"]
    	#self.pages = ["SubjectInfo.htm"]
        self.savedir = os.path.dirname(globals()["__file__"])			# Directory where users' answers are saved
        ## Screen settings
        self.geometry=[100,100,1000,800] # Position and size of window
        self.overwriteFiles = 0          # if 1 overwrites savefiles, otherwise numbers them
        self.showLoadButton = 0
        self.showPreviousButton = 1
        self.showReloadButton = 1
        self.fullscreen = 0
        
   
    def pre_mainloop(self):
        # If absolute paths are not given, make them relative to module
        path = os.path.join(os.path.dirname(globals()["__file__"]),'pages')
        for ii in range(len(self.pages)):
            if not os.path.isabs(self.pages[ii]):
                self.pages[ii] = os.path.join(path,self.pages[ii])
        
        # If pages is not a list, make a list
        if isinstance(self.pages,str):
            self.pages = [self.pages]
        self._nPages = len(self.pages)
        self._currentPage = 0
        self._isform = [False]*self._nPages          # Whether a page is a form
        self._submitted = [False]*self._nPages          # Holds whether a form was already submitted        
        
        # Initialize the applicationm, frame and panels
        self.app = wx.App(False)    # Create a new app, don't redirect stdout/stderr to a window.
        if self.fullscreen:
            self.geometry[2:] = wx.DisplaySize()

        # create a window/frame, no parent, -1 is default ID, title, size
        self.frame = wx.Frame(None, -1, "HTMLViewer", pos=wx.Point(self.geometry[0],self.geometry[1]), size=self.geometry[2:])
        # button panel
        self.buttonPanel = ButtonPanel(self,self.frame,-1,self.geometry)
        ## Make HTML panels
        self.htmlPanels = []
        for ii in range(self._nPages):
            HTMLforms.ISFORM = 0
            newpanel = HTMLViewerPanel(self,self.frame,-1,self.geometry,self.pages[ii])
            #self.newpanel = HTMLViewerPanel(self,self.frame,-1,self.geometry,self.pages[ii])
            self._isform[ii] = HTMLforms.ISFORM
            if ii>0:
                newpanel.Hide()
            self.htmlPanels.append(newpanel)

        ## Make transient popup window
        self.pop = TransientPopup(self.frame,"Data saved")


        ## Show window
        if self.fullscreen:
            self.frame.ShowFullScreen(True)
        else:
            self.frame.Show()
        #self.frame.Show(True)
        self.updateHTML()
        self.app.MainLoop()          # start the event loop

        
    def post_mainloop(self):
        pass

    def updateHTML(self):
        # Hide all panels
        for ii in range(self._nPages):
            self.htmlPanels[ii].Hide()
        # 
        self.htmlPanels[self._currentPage].Show()
        
        ## Set Frame title
        #self.htmlPanels[self._currentPage].SetRelatedFrame(self.parent,"%s")
        self.buttonPanel.pagelabel.SetLabel("Page "+str(self._currentPage+1)+"/"+str(self._nPages))
        ## Disable Next button if its a form that was not submitted yet
        if self._isform[self._currentPage] and not self._submitted[self._currentPage]:
            self.buttonPanel.btnNext.Enable(False)
            self.buttonPanel.btnClose.Enable(False)

    def save(self,filename,args):
        ''' Saves questionnaire results to file '''
        fullfile = os.path.join(self.savedir,filename)
        if not self.overwriteFiles and os.path.isfile(fullfile):
            parts = filename.split('.')
            nr = 2
            while os.path.isfile(fullfile):
                fullfile = os.path.join(self.savedir,parts[0]+str(nr)+'.'+parts[1])
                nr += 1
        f = open(fullfile, 'w')
        # Print header with date
        import datetime
        f.write('# '+str(datetime.datetime.today())+'\n')
        # Print data
        for key in args.keys():
            f.write(key+"="+args[key]+"\n")
        f.close()
        self._submitted[self._currentPage] = True
        self.buttonPanel.NextPage(None)
        self.pop.Popup()
        self.pop.timer.Start(2000)

###############################
#    Panel with buttons       #
###############################
class ButtonPanel(wx.Panel):
    """
    The panel wherein the top row of buttons is shown
    """
    def __init__(self, view, parent, idd, geometry):
        # default pos is (0, 0) and size is (-1, -1) which fills the frame

        self.view = view
        ## Graphical layout
        wx.Panel.__init__(self, parent, idd,pos=(0,0),size=(geometry[2],30))
        self.SetBackgroundColour("gray")
        self.parent = parent

        self.pagelabel = wx.StaticText(self, -1, "Page "+str(self.view._currentPage+1)+"/"+str(self.view._nPages) , wx.Point(10, 6))
        xpos = 100
        xoff = 100
        
        if self.view.showLoadButton:
            self.btn1 = wx.Button(self, -1, "Load Html File", pos=(xpos,0))
            self.btn1.Bind(wx.EVT_BUTTON, self.OnLoadFile)
            xpos += xoff
     
        if self.view.showReloadButton:
            self.btnReload = wx.Button(self, -1, "Reload", pos=(xpos,0))
            self.btnReload.Bind(wx.EVT_BUTTON, self.OnReloadFile)
            xpos += xoff

        if self.view.showPreviousButton:
            self.btnPrevious = wx.Button(self, -1, "Previous", pos=(xpos,0))
            self.btnPrevious.Bind(wx.EVT_BUTTON, self.PreviousPage)
            self.btnPrevious.Enable(False)
            xpos += xoff

        self.btnNext = wx.Button(self, -1, "Next", pos=(xpos,0))
        self.btnNext.Bind(wx.EVT_BUTTON, self.NextPage)
        xpos += xoff

        self.btnClose = wx.Button(self, -1, "Close", pos=(xpos,0))
        self.btnClose.Bind(wx.EVT_BUTTON, self.OnClose)
               
        #self.updateHTML()   

    def PreviousPage(self, event):
        if self.view._currentPage>0:
            self.view._currentPage -= 1
        self.btnPrevious.Enable(self.view._currentPage > 0)
        self.btnNext.Enable(self.view._currentPage < self.view._nPages-1)
        self.btnClose.Enable(True)
        self.view.updateHTML()
    
    def NextPage(self, event):
        if self.view._currentPage<self.view._nPages-1:
            self.view._currentPage += 1
        if self.view.showPreviousButton:
            self.btnPrevious.Enable(self.view._currentPage > 0)
        self.btnNext.Enable(self.view._currentPage < self.view._nPages-1)
        self.btnClose.Enable(True)
        self.view.updateHTML()
       
    ''' HTML Event handlers '''    
    def OnLoadFile(self, event):
        dlg = wx.FileDialog(self, wildcard = '*.htm*', style=wx.OPEN)
        if dlg.ShowModal():
            path = dlg.GetPath()
            self.html.LoadPage(path)
        dlg.Destroy()
    
    def OnReloadFile(self, event):
        self.view.htmlPanels[self.view._currentPage].reload()

    def OnClose(self, event):
        self.parent.Close()


###############################
#   Panel with html file      #
###############################
class HTMLViewerPanel(wx.Panel):

    """
    The panel wherein the HMTL page is contained.
    """
    def __init__(self, view, parent, idd, geometry,page):
        # default pos is (0, 0) and size is (-1, -1) which fills the frame
        self.page = page
        self.view = view
        ## Graphical layout
        wx.Panel.__init__(self, parent, idd,pos=(0,30),size=geometry[2:])
        self.SetBackgroundColour("blue")
        self.parent = parent
        self.html = wx.html.HtmlWindow(self, idd, pos=(0,0), size=geometry[2:],style= wx.html.HW_DEFAULT_STYLE | wx.TAB_TRAVERSAL)
        self.html.Bind(HTMLforms.EVT_FORM_SUBMIT, self.OnFormSubmit)
        self.html.LoadPage(page)
    
    def reload(self):
        self.html.LoadPage(self.page)
        


                
    def OnFormSubmit(self,evt):
        ''' Pressing the submit button in an HTML form '''
        self.view.save(evt.form.action,evt.args)

class TransientPopup(wx.PopupTransientWindow):
    def __init__(self, parent,text):
        wx.PopupTransientWindow.__init__(self, parent, wx.NO_BORDER)
        panel = wx.Panel(self, -1)		
        panel.SetBackgroundColour("GREEN")
        st = wx.StaticText(panel, -1,text, pos=(10,10))
        sz = st.GetBestSize()
        panel.SetSize( (sz.width+20, sz.height+20) )
        self.SetSize(panel.GetSize())
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
    
    def OnTimer(self,event):
        self.Dismiss()
        self.timer.Stop()


if __name__ == '__main__':
    html = HTMLViewer()
    html.init()
    html.pre_mainloop()
    html.post_mainloop()
