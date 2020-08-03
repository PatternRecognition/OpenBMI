
#!/usr/bin/env python

# marton danoczy, jan 2011
# code adapted from http://docs.python.org/library/socketserver.html

import socket
import threading
import SocketServer
import win32com.client
import pythoncom
import shutil
import os
from hashlib import md5
from base64 import urlsafe_b64encode
from time import sleep

HOST     = "sony3d.ml.tu-berlin.de"
PORT     = 12345
BUFSIZ   = 2048
wav      = "D:\\project_sony3d\\blank.wav"
temp_dir = "D:\\project_sony3d\\temp"

class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):
    def setup(self):
        print "starting thread %s" % threading.currentThread()
        self.ok = False
        pythoncom.CoInitialize()
        self.player = win32com.client.Dispatch("StereoPlayer.Automation")
        self.wait()
        f = "D:\\101215_Cube\\black_l.png"
        self.player.OpenLeftRightFiles(f, f, wav, 1)
        self.wait()
        self.player.EnterFullscreenMode(True)
        self.wait()
        self.ok = True
    
    def wait(self):
        ready = False
        while not ready:
            ready = self.player.GetReady()

    def process(self, s):
        if len(s)==0:
            return
        cmd, param = s.split(' ', 1)
        if cmd == 'pic':
            self.show_pic(param)
        elif cmd == 'jpg':
            self.show_jpg(param)
        elif cmd == 'msg':
            self.show_msg(param)
        elif cmd == 'vid':
            self.show_vid(param)

    def show_pic(self, fname):
        self.ok = False;
        f1 = "D:\\101215_Cube\\%s_l.png" % fname
        f2 = "D:\\101215_Cube\\%s_r.png" % fname
        print "Opening: %s" % fname
        # If file names are too long, COM fails. Workaround: copy files
        hash = urlsafe_b64encode(md5(fname).digest())
        t1 = "D:\\project_sony3d\\temp\\%s_l.png" % hash;
        t2 = "D:\\project_sony3d\\temp\\%s_r.png" % hash;
        shutil.copyfile(f1, t1)
        shutil.copyfile(f2, t2)
        self.player.OpenLeftRightFiles(t1, t2, wav, 1)
        self.wait()
        self.ok = True;

    def show_msg(self, fname):
        self.ok = False;
        f = "D:\\project_sony3d\\%s.png" % fname
        print "Opening: %s" % fname
        self.player.OpenLeftRightFiles(f, f, wav, 1)
        self.wait()
        self.ok = True;
        
    def show_jpg(self, fname):
        self.ok = False;
        f = "D:\\project_sony3d\\%s.jpg" % fname
        print "Opening: %s" % fname
        self.player.OpenLeftRightFiles(f, f, wav, 1)
        self.wait()
        self.ok = True;

    def show_vid(self, s):
        self.ok = False;
        try:
            secs, fname = s.split(' ', 1)
            fvid = "D:\\project_sony3d\\%s" % fname
            print "Opening: %s" % fname
            self.player.OpenLeftRightFiles(fvid, fvid, "", 2)
            self.player.SetPosition(float(secs))
            self.player.SetPlaybackState(0)
        except ValueError:
            self.player.SetPlaybackState(1)
        self.wait()
        self.ok = True;
    
    def handle(self):
        while self.ok:
            data = self.request.recv(BUFSIZ)
            if not data:
                break
            else:
                self.process(data.strip())

    def finish(self):
        print "stopping thread %s" % threading.currentThread()
        
class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

if __name__ == "__main__":

    server_set_up = False
    while not server_set_up:
        try:
            server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
            server_set_up = True
        except socket.gaierror:
            print "Couldn't start listening: maybe cable isn't plugged in?"
            sleep(2)
        
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)

    # Exit the server thread when the main thread terminates
    server_thread.setDaemon(True)
    server_thread.start()

    try:
        os.mkdir(temp_dir)
    except:
        pass
    
    print "Server is up and running on %s:%i" % (HOST, port) 

    try:
        print "Press CTRL+C to shutdown."
        while True:
            pass
    except:
        print

    print "Shutting down..."
    server.shutdown()
 
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print "done."