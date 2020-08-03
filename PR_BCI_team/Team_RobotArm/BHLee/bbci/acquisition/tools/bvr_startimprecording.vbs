Set objArgs = WScript.Arguments
Set Rec = CreateObject("VisionRecorder.Application")
Rec.Acquisition.ViewImpedance()
WScript.Sleep 10000
Rec.Acquisition.ViewData()
Rec.Acquisition.StartRecording(objArgs(0))
